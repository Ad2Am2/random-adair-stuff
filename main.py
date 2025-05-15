import importlib.util
import sys
import getpass
from assistant_role import Pilot, ATC
from cpdlc_translation import (
    VLLM_Model,
    OpenAI_Model,
    RAG_CPDLCRetriver,
    LLM_CPDLCRetriver,
    CPDLCTranslater,
    TranslationLogger
)

# ----- Default Configurations -----
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 40
DEFAULT_GEN_PARAMS = {
    "temperature" : 0.2,
    "top_p" : 0.9,
    "max_tokens" : 512,
    "repetition_penalty" : 1.00
}

# ----- Functions to validate the requirements -----
PACKAGE_IMPORT_NAMES = {"faiss-cpu" : "faiss", "sentence-transformers" : "sentence_transformers"}

def validate_reqs() -> tuple[bool, list[str]]:
    """Ensures all required packages are installed by checking requirements.txt"""
    packages_missing = False
    reqs_missing = []
    with open('requirements.txt') as requirements:
        for package in requirements:
            package = package.replace("\n", "").strip()
            package_found = importlib.util.find_spec(get_package_name(package)) is not None

            if not package_found:
                if not packages_missing:
                    print("Packages missing:")
                    packages_missing = True

                print(package)
                reqs_missing.append(package)

    return packages_missing, reqs_missing

def get_package_name(package : str) -> str:
    if package in PACKAGE_IMPORT_NAMES:
        return PACKAGE_IMPORT_NAMES[package]
    
    return package


# ---- Helper Functions -----
# Header function for consistency of UI
def print_header(title):
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)

# Gets a valid choice from the user 
def get_choice(prompt, options):
    while True:
        print(prompt)
        for key, value in options.items():
            print(f"\t{key}. {value}")
        choice = input("Enter your choice: ").strip()
        if choice in options:
            return choice
        else:
            print("Invalid choice. Please try again.")

# Gets input from the user with optional default, requirement, type, and validation.
def get_input(prompt, default=None, required=False, input_type=str, validator=None):
    while True:
        display_prompt = f"{prompt}"
        if default is not None:
            display_prompt += f" [{default}]"
        display_prompt += ": "
        value_str = input(display_prompt).strip()
        if not value_str and default is not None:
            value_str = str(default)

        if not value_str and required:
            print("This field is required.")
            continue

        if not value_str and not required:
            return None

        try:
            value = input_type(value_str)
            if validator and not validator(value):
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a value of type {input_type.__name__}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def get_yes_no(prompt, default_yes=True):
    default_str = "[Y/n]" if default_yes else "[y/N]"
    while True:
        choice = input(f"{prompt} {default_str}: ").strip().lower()
        if not choice:
            return default_yes
        if choice in ['y', 'yes']:
            return True
        if choice in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'.")

# ----- Configuration Steps -----
# Function to select the user role (Pilot or ATC)
def select_user_role(app_config):
    print_header("Step 1: Select User Role")
    options = {'1': 'Pilot (Translating Pilot Messages)', '2': 'ATC (Translating ATC Messages)'}
    choice = get_choice("Choose the source of the messages to be translated:", options)
    if choice == '1':
        app_config['role_class'] = Pilot
        app_config['role_name'] = "Pilot"
    else:
        app_config['role_class'] = ATC
        app_config['role_name'] = "ATC"
    print(f"User role set to: {app_config['role_name']}")

# Selects the LLM provider (vLLM or OpenAI)
def select_model_provider(app_config, step=2):
    print_header(f"Step {step}: Select Model Provider")
    options = {'1': 'vLLM (Local Execution)', '2': 'OpenAI API (Server-based)'}
    choice = get_choice("Choose how the LLM will be run:", options)
    app_config['model_provider'] = "vLLM" if choice == '1' else "OpenAI"
    print(f"Model provider set to: {app_config['model_provider']}")

# Configures the specific model and its parameters.
def configure_model(app_config, step=3):
    print_header(f"Step {step}: Configure Language Model")
    provider = app_config['model_provider']
    model_config = {}
    gen_params_config = {}

    # Model Name and Params Setup
    if provider == "vLLM":
        # Choose to use automatic/recommended settings or manual
        options = {'1': 'Automatic (Recommended Model and Parameters)', '2': 'Manual Configuration'}
        choice = get_choice("Choose how to setup the large language model:", options)
        automatic = choice == '1'

        if automatic:
            model_name = DEFAULT_VLLM_MODEL
            model_config['quantization'] = None
            model_config['tensor_parallel_size'] = 1
        else:
            model_name = get_input("Enter vLLM model name", default=DEFAULT_VLLM_MODEL, required=True)
            q_options = {'1': 'None', '2': 'AWQ (4-bit)', '3': 'BitsAndBytes (4-bit)'}
            q_choice = get_choice("Choose Manual Quantization:", q_options)
            model_config['quantization'] = {'1': None, '2': 'awq', '3': 'bitsandbytes'}[q_choice]
            model_config['tensor_parallel_size'] = get_input(
                "Enter tensor parallel size (e.g., 1)", default=1, input_type=int,
                validator=lambda x: x >= 1 or print("Must be >= 1")
            )
    else:
        model_name = get_input("Enter OpenAI model name", default="gpt-4o", required=True)
        # Securely get API key
        try:
            api_key = getpass.getpass(prompt="Enter your OpenAI API Key: ")
            if not api_key:
                print("API Key is required for OpenAI.")
                sys.exit(1)
            model_config['api_key'] = api_key
        except Exception as e:
            # getpass might fail in some environments
            print(f"\nError getting API key: {e}")
            model_config['api_key'] = get_input("Enter your OpenAI API Key (visible input):", required=True)

    app_config['model_name'] = model_name
    app_config['model_config'] = model_config

    # Choose how to setup parameters
    param_choice = get_choice(
        "Configure generation parameters:",
        {'1': 'Automatic (Use Defaults)', '2': 'Advanced (Customize)'}
    )

    if param_choice == '1':
        gen_params_config = DEFAULT_GEN_PARAMS
        print("Using default generation parameters.")
    else:
        print("\n--- Advanced Generation Parameter Configuration ---")
        current_params = DEFAULT_GEN_PARAMS.copy()
        for key, default_value in DEFAULT_GEN_PARAMS.items():
            input_type = int if isinstance(default_value, int) else float if isinstance(default_value, float) else str
            validator = None
            if key == 'temperature':
                 validator=lambda x: 0.0 <= x <= 2.0 or print("Temperature must be between 0.0 and 2.0")
            elif key == 'top_p':
                 validator=lambda x: 0.0 <= x <= 1.0 or print("Top_p must be between 0.0 and 1.0")
            elif key == 'max_tokens':
                 validator=lambda x: x > 0 or print("Max_tokens must be positive")
            elif key == 'repetition_penalty':
                 validator=lambda x: x > 0 or print("Repetition penalty must be positive")

            new_value = get_input(f"Enter value for '{key}'", default=default_value, input_type=input_type, validator=validator)
            current_params[key] = new_value
        gen_params_config = current_params
        print("Advanced generation parameters set.")

    app_config['gen_params'] = gen_params_config

# Configures the retrieval system (RAG or LLM).
def configure_retrieval(app_config):
    print_header("Step 4: Configure Retrieval System")
    retriever_config = {}

    options = {'1': 'RAG (Recommended)', '2': 'Another LLM (NOT COMPLETED)', '3': 'None'}
    choice = get_choice("Choose the retrieval system for CPDLC message context:", options)

    if choice == '1': # Rag
        app_config['retriever_type'] = "RAG"
        param_choice = get_choice(
            "Configure RAG parameters:",
            {'1': 'Automatic (Recommended)', '2': 'Advanced (Custom)'}
        )
        if param_choice == '1':
            retriever_config['embedder_model'] = DEFAULT_EMBEDDER
            retriever_config['k'] = DEFAULT_TOP_K
            print("Using default RAG parameters.")
        else:
            print("\n--- Advanced RAG Parameter Configuration ---")
            retriever_config['embedder_model'] = get_input(
                "Enter Sentence Transformer embedder model name",
                default=DEFAULT_EMBEDDER, required=True
            )
            retriever_config['k'] = get_input(
                "Enter number of messages to retrieve (k)",
                default=DEFAULT_TOP_K, required=True, input_type=int,
                validator=lambda x: x > 0 or print("k must be positive")
            )
            print("Advanced RAG parameters set.")
    # TODO: This option does not work yet as prompts need to be added for LLM retrieval and safeguards
    elif choice == '2':
        app_config['retriever_type'] = "LLM"
        print("Using the main LLM for retrieval tasks (configuration not needed here).")
    else:
        app_config['retriever_type'] = None
        print("No retrieval system will be used.")

    app_config['retriever_config'] = retriever_config

# Initializes backend components based on configuration.
def initialize_components(app_config) -> CPDLCTranslater:
    print("\n--- Initializing Components ---")
    try:
        # 1. Initialize User Role
        user_role = app_config['role_class']()
        print(f"Initialized User Role: {app_config['role_name']}")

        # 2. Initialize Model
        model = None
        if app_config['model_provider'] == "vLLM":
            quantization_method = app_config['model_config']['quantization']
            if quantization_method == 'awq':
                if importlib.util.find_spec('autoawq') is None:
                    quantization_method = 'bitsandbytes'
                    print('Packakge autoawq not installed. Switched to bitsandbytes quantization')
                
            # Actual vLLM initialization
            try:
                 num_gpus = 1
                 if 'tensor_parallel_size' in app_config['model_config']:
                      num_gpus = app_config['model_config']['tensor_parallel_size']

                 # Pass model_config and gen_params in init
                 model = VLLM_Model(
                     model_name=app_config['model_name'],
                     model_config={'quantization': quantization_method, 'tensor_parallel_size': num_gpus},
                     gen_params=app_config['gen_params']
                 )
                 print(f"\nInitialized vLLM Model.")
            except Exception as e:
                 print(f"\n--- ERROR Initializing vLLM Model ---")
                 print(f"Model: {app_config['model_name']}")
                 print(f"Config used: {{'quantization': {quantization_method}, 'tensor_parallel_size': {num_gpus}}}")
                 print(f"Error: {e}")
                 print("Check model name/path, available VRAM, and CUDA setup.")
                 print("\nExiting Application.")
                 sys.exit(1)

        elif app_config['model_provider'] == "OpenAI":
            model = OpenAI_Model(
                model_name=app_config['model_name'],
                model_config=app_config['model_config'],
                openai_api_key=app_config['model_config']['api_key'],
                gen_params=app_config['gen_params']
            )
            print(f"\nInitialized OpenAI Model.")

        # 3. Initialize Retriever
        retriever = None
        if app_config['retriever_type'] == "RAG":
            retriever = RAG_CPDLCRetriver(
                cpdlc_data=user_role.get_cpdlc_data(),
                embedder_name=app_config['retriever_config']['embedder_model'],
                k=app_config['retriever_config']['k']
            )
            print(f"\nInitialized RAG Retriever.")
        elif app_config['retriever_type'] == "LLM":
            # TODO: NOT DONE
            raise NotImplementedError
            retriever = LLM_CPDLCRetriver(
                cpdlc_data=user_role.get_cpdlc_data(),
                model=model)
            print("\nInitialized LLM Retriever (using main model)")
        else:
            print("No retriever system selected.")


        # 4. Initialize Translator
        translator = CPDLCTranslater(
            user=user_role,
            model=model,
            retriever=retriever
        )
        print("\nInitialized CPDLC Translator.")
        return translator

    except Exception as e:
        print(f"\n--- ERROR during component initialization ---")
        print(f"Error: {e}")
        print("Please check your configuration and backend code.")
        sys.exit(1)

def write_logs_to_file(logger : TranslationLogger):
    logs_str = logger.get_print_ready()

# ----- Main Application Logic -----
def run_application():
    count_conversations = 0
    app_config = {}

    # Configuration Steps
    select_user_role(app_config)
    select_model_provider(app_config)
    configure_model(app_config)
    configure_retrieval(app_config)

    # Initialize Backend
    translator = initialize_components(app_config)

    # Interaction Loop
    print_header("Step 5: Process Messages")


    # Ask if logs are to be kept
    logger = None
    keep_logs = get_yes_no("Do you want to keep all logs of execution?", default_yes=False)
    if keep_logs:
        logger = TranslationLogger()

    # Ask for conversation history
    conversation_id = None
    use_conversation = get_yes_no("Do you wish to use messages as part of a conversation (stores them for model messaging context history)?", default_yes=False)
    if use_conversation:
        conversation_id = get_input("Enter a Conversation ID (or leave blank for automatic)", required=False)
        if not conversation_id:
            count_conversations += 1
            conversation_id = f"CID_{count_conversations}"
        print(f"Using Conversation ID: {conversation_id}")
        print("\nTo change to a new conversation in program loop below, type 'change'.")

    store_history_flag = use_conversation
    while True:
        print("\nEnter your natural language message below (or type 'exit' to quit)")
        natural_language_input = input("> ").strip()

        if store_history_flag and natural_language_input.lower() == 'exit':
            break
        
        if natural_language_input.lower() == 'change':
            conversation_id = get_input("Enter a Conversation ID (or leave blank for automatic)", required=False)
            if not conversation_id:
                count_conversations += 1
                conversation_id = f"CID_{count_conversations}"
            print(f"Using Conversation ID: {conversation_id}")
            continue
        
        print("--- Processing ---")
        try:
            response = translator.translate(
                natural_language=natural_language_input,
                keep_history=store_history_flag,
                recipient_id=conversation_id,
                logger=logger
            )

            print("\n--- Translation Result ---")
            if response and 'message' in response and 'context' in response:
                print(f"- Instruction: {response['message']}")
                print(f"- Context: {response['context']}")
                if 'reference' in response:
                    print(f"- Ref #: {response['reference']}")
            else:
                print("Processing failed or returned empty result.")
            print("------------------------")

        except Exception as e:
            print(f"\n--- ERROR during message processing ---")
            print(f"Error: {e}")

    if logger is not None:
        file_path = write_logs_to_file(logger)
        print(f"\nWrote logs to file path: {file_path}")

    translator.clear_memory()
    print("\nExiting application. Goodbye!")


if __name__ == "__main__":
    # Validate all packages are installed
    try:
        packages_missing, _ = validate_reqs()
        if packages_missing:
            print("\nPlease install the missing packages listed above to run this application.\nExiting application.")
            sys.exit(1)
    except OSError or ValueError:
        print("\nUnexpected error occured while validating the requirements.\nExiting application.")
        sys.exit(1)
    
    # Run the application
    run_application()