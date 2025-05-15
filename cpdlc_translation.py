from transformers import AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
import faiss
import numpy as np
import torch
import json
import os
import gc
from openai import OpenAI
from assistant_role import User

# ----- Utilities -----
class TranslationLogger:
    def __init__(self):
        self.__logs = []

    def log(self, input, llm_response, is_success, rag_retrived_messages, expected_output) ->  None:
        self.__logs.append((input, llm_response, is_success, rag_retrived_messages, expected_output))

    def get_logs(self) -> list:
        return self.__logs
    
    def get_print_ready(self) -> str:
        logs_str = ""
        count = 1
        for input, llm_response, is_success, rag_retrived_messages, expected_output in self.__logs:
            logs_str += f"------------- Log {count} -------------"
            logs_str += f"\nInput:                 {input}"
            logs_str += f"\nLLM Response:          {llm_response}"
            if expected_output != "":
                logs_str += f"\nExpected Response:     {expected_output}"
            logs_str += f"\nValid Response:        {is_success}"
            logs_str += f"\nRetrieved Messages     {rag_retrived_messages}"
            count += 1

        return logs_str


# ----- Quantizations -----
class QuantizedConfigs:
    @staticmethod
    def get_4bit_bnb() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    
    # Currently not used as VLLM only supports default bitsandbytes str arg (produces 4bit quant)
    @staticmethod
    def get_8bit_bnb() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(load_in_8bit=True)
    
# ----- Large Language Models Classes (OpenAI vs vLLM) -----
class Model:
    def __init__(self, model_name : str, model_config : dict, gen_params : dict = None):
        self.model_name = model_name
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = gen_params
        self._load_model(model_config)

    def _load_model(self, model_config) -> None:
        raise NotImplementedError

    def generate(self, prompt, gen_params : dict = None) -> str:
        raise NotImplementedError
    
    def clear_memory(self) -> None:
        raise NotImplementedError
    
    def get_gen_params(self) -> dict:
        return self.gen_params.copy()
    
    def get_model_name(self) -> str:
        return self.model_name

    @staticmethod
    def get_default_gen_params() -> dict:
        return {"temperature" : 0.2, "top_p" : 0.9, "max_tokens" : 512}

class VLLM_Model(Model):
    def __init__(self, model_name : str, model_config : dict, gen_params : dict = None) -> None:
        super().__init__(model_name, model_config)
        self.gen_params = gen_params if gen_params is not None else Model.get_default_gen_params()
        self.sampling_params = SamplingParams(**self.gen_params)

    def _load_model(self, model_config) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        os.environ['VLLM_USE_V1'] = '0' # To counter bug within vLLM module
        self.model = LLM(model=self.model_name, **model_config, enforce_eager=True, disable_async_output_proc=True)

    def generate(self, messages, gen_params : dict = None) -> str:
        params = self.sampling_params if gen_params is None else SamplingParams(**gen_params)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.model.generate([text], sampling_params=params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        return response

    def clear_memory(self) -> None:
        del self.tokenizer
        del self.model

class OpenAI_Model(Model):
    def __init__(self, model_name : str, model_config : dict, openai_api_key, openai_api_base : str = "http://localhost:8000/v1", gen_params : dict = None) -> None:
        super().__init__(model_name, model_config)
        self.gen_params = gen_params if gen_params is not None else Model.get_default_gen_params()
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def _load_model(self, model_config) -> None:
        pass

    def generate(self, messages, gen_params : dict = None) -> str:
        params = self.gen_params if gen_params is None else gen_params
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
        return response

    def clear_memory(self) -> None:
        self.client.close()

# ----- CPDLC Retrieval Classes (RAG vs LLM) -----
# Base Class of retrievals for which, if instantiated, all CPDLC messages are used in context
class CPDLCRetriever:
    WORD_TYPES = {
        "level" : "level",
        "reach" : "level",
        "above" : "level",
        "below" : "level",
        "between" : "level/time",
        "climb" : "level/altitude",
        "climbing" : "level/altitude",
        "descent" : "level/altitude",
        "descend" : "level/altitude",
        "descending" : "level/altitude",
        "maintain" : "level/altitude/speed",
        "maintaining" : "level/altitude/speed",
        "at" : "time/rate/level/speed",
        "by" : "time",
        "before": "time",
        "after" : "time",
        "less" : "speed",
        "great" : "speed",
        "offset" : "specified distance",
        "contact" : "time",
        "deviate" : "specified distance",
        "deviation" : "specified distance",
        "turn" : "degrees",
        "heading" : "degrees",
        "expect" : "speed/level",
        "speed" : "speed",
        "exceed" : "speed",
        "contact" : "frequency",
        "frequency" : "frequency",
        "monitor" : "frequency",
        "block" : "level",
        "leaving" : "level",
        "leave" : "level",
        "accept" : "speed/level/distance",
        "microphone" : "frequency",
        "time" : "time",
        "track" : "degrees"
    }

    def __init__(self, cpdlc_data) -> None:
        self.cpdlc_data = cpdlc_data
        self.k = None

    def retrieve_cpdlc(self, natural_language : str, k : int = 40) -> list:
        parsed_message = self.__process_input_for_rag(natural_language)
        return self._retrieve_top_k(parsed_message, k)
    
    def clear_memory(self):
        raise NotImplementedError
    
    def _retrieve_top_k(self, message : str, k : int) -> list:
        return self.cpdlc_data

    def __is_word_digit(self, word : str) -> bool:
        if len(word) == 0:
            return False
        for c in word:
            if not (c.isdigit() or c == '.' or c == ":"):
                return False
        return True

    def __get_numbers_indices(self, words : list[str]) -> list[int]:
        indices = []
        for idx, word in enumerate(words):
            if self.__is_word_digit(word):
                indices.append(idx)
        return indices

    def __process_input_for_rag(self, message : str) -> str:
        words = message.split()
        modified = False
        if 'direct' in words:
            direct_idx = words.index('direct')
            words_len = len(words)
            if direct_idx + 1 < words_len and words[direct_idx + 1] == 'to':
                words = words[:direct_idx + 2] + ['position'] + (words[direct_idx + 2 : ] if direct_idx + 2 < words_len else [])
            else:
                words = words[:direct_idx + 1] + ['position'] + (words[direct_idx + 1 : ] if direct_idx + 1 < words_len else [])

            modified = True

        numbers = self.__get_numbers_indices(words)
        if len(numbers) == 0 and not modified:
            return message
        
        # Make the changes
        i = 0
        last = None
        for r in numbers:
            while i < r:
                if words[i] in self.WORD_TYPES:
                    last = words[i]
                i += 1
            if ':' in words[r]:
                words[r] = "time"
            elif last is not None:
                words[r] = self.WORD_TYPES[last]
            i += 1
        return " ".join(words).strip()


# CPDLC Retriver using RAG
class RAG_CPDLCRetriver(CPDLCRetriever):
    def __init__(self, cpdlc_data : list, embedder_name : str, k = None) -> None:
        super().__init__(cpdlc_data)
        self.embedder_name = embedder_name
        self.embedder = SentenceTransformer(embedder_name)
        self.__load_rag()
        self.k = k

    # Function to initialize the RAG mechanism with faix indexes, it uses the CPDLC message element as embeddings
    def __load_rag(self) -> None:
        # Create embeddings for all CPDLC messages
        text_entries = [
            item['Message_Element'].replace('[', '').replace(']', '').lower()
            for item in self.cpdlc_data
        ]
        embeddings = self.embedder.encode(text_entries)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    # Function to retrieve top corresponding cpdlc message elements
    def _retrieve_top_k(self, message : str, k : int = 40) -> list:
        query_embedding = self.embedder.encode([message])
        _, indices = self.index.search(query_embedding, k)
        return [self.cpdlc_data[i] for i in indices[0]]
    
    def clear_memory(self):
        del self.embedder
        del self.index


# CPDLC Retriver using a LLM (Still needs further developments)
class LLM_CPDLCRetriver(CPDLCRetriever):
    def __init__(self, cpdlc_data : list, model : Model, k = None) -> None:
        super().__init__(cpdlc_data)
        self.model = model
        self.k = k

    # Function to retrieve top corresponding cpdlc message elements
    def _retrieve_top_k(self, message : str, k : int = 40) -> list:
        response = self.model.generate(message)
        return response.split("\n")[:k]
    
    def clear_memory(self):
        self.model.clear_memory()


# ----- Main Translater Class -----
class CPDLCTranslater:
    def __init__(self, user : User, model : Model, retriever : CPDLCRetriever, additional_context : str = "") -> None:
        self.user = user
        self.model = model
        self.retriever = retriever
        self.additional_context = additional_context
        self.message_history = {}
        self.available_temps = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def change_model(self, model : Model, additional_context : str = None) -> None:
        self.model = model
        self.additional_context = additional_context

    def change_retriever(self, retriever : CPDLCRetriever) -> None:
        self.retriever = retriever

    def add_recipient(self, recipient_id) -> None:
        self.message_history[recipient_id] = []

    def parse_llm_response(self, llm_response : str, prompt_end : str = None) -> tuple[str, bool]:
        json_str = "```json"
        if json_str in llm_response:
            llm_response = llm_response[llm_response.index(json_str):]

        if '{' not in llm_response or '}' not in llm_response:
            return "", False

        if '[' in llm_response or ']' in llm_response:
            return llm_response, False

        response = llm_response[llm_response.index('{'): llm_response.index('}') + 1]
        return response, True

    def translate(self, natural_language, logger : TranslationLogger = None, keep_history = False, recipient_id = None, k=40) -> dict:
        # Retrieve top cpdlc corresponding messages as context
        relevant_messages = self.retriever.retrieve_cpdlc(natural_language, k)

        # Format the cpdlc descriptors
        context_str = "\n".join(
            f"CPDLC Message: '{msg['Message_Element']}'. Intent: '{msg['Message_Intent']}'. Reference Number: '{msg['Ref_Num']}'."
            for msg in relevant_messages
        )

        # Format context and prepare input for llm
        system_prompt = self.user.get_system_prompt()
        prompt = system_prompt.format(context=context_str)
        end_model_input = "Input message:\n" + natural_language + self.additional_context + "\n"

        # Create the input for llm based on history or single message
        message = {"role": "user", "content": prompt + end_model_input}
        if keep_history and recipient_id is not None:
            if recipient_id not in self.message_history:
                self.message_history[recipient_id] = []
            self.message_history[recipient_id].append(message)
            messages = self.message_history[recipient_id]
        else:
            messages = [message]

        # Loop while no valid response is given or count reaches 5
        missing_valid_response = True
        response_str = ""
        count = 0
        gen_params = None
        while missing_valid_response:
            # Get response
            response_str_temp = self.model.generate(messages, gen_params)
            response_str, success = self.parse_llm_response(response_str_temp, end_model_input)
            missing_valid_response = not success

            # Logging if needed
            if logger is not None:
                logger.log(natural_language, response_str, success, context_str, "")

            # Handle case where the response is not valid
            if missing_valid_response:
                temperature = np.random.choice(self.available_temps)
                gen_params = self.model.get_gen_params()
                gen_params['temperature'] = temperature
                count += 1
                if count > 5:
                    print("\nLLM failed to generate valid response.\n")
                    return {}

        if keep_history and recipient_id is not None:
            assistant_msg = {"role": "assistant", "content": response_str}
            self.message_history[recipient_id].append(assistant_msg)

        try:
            response_json = json.loads(response_str)
            return response_json
        except json.JSONDecodeError:
            print("\nFailed to parse json for llm response.\n")
            return {}
        
    def clear_memory(self):
        self.model.clear_memory()
        self.retriever.clear_memory()
        gc.collect()