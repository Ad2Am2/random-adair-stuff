import json
import sys

# Assumes current path
CPDLC_DOWNLINK_PATH = "CPDLC_Downlink.json"
CPDLC_UPLINK_PATH = "CPDLC_Uplink.json"

# 'Abstract' Class that provides confined constants for different purposes
class User:
    def __init__(self):
        self.cpdlc_data = []

    def get_cpdlc_data(self) -> list:
        raise NotImplementedError
    
    def get_system_prompt(self) -> str:
        raise NotImplementedError
    
    # Loads CPDLC data from parent class
    def _load_data(self, path):
        try:
            with open(path, "r") as cpdlc_json:
                self.cpdlc_data = json.load(cpdlc_json)
        except json.JSONDecodeError:
            print(f"Error occured when reading the CPDLC json data from path '{path}'. Error Description:\n")
            print(sys.exc_info()[0])

# When a Pilot uses the application
class Pilot(User):
    def __init__(self, cpdlc_data_path=None) -> None:
        super().__init__()
        self.path = CPDLC_DOWNLINK_PATH if cpdlc_data_path is None else cpdlc_data_path
        self._load_data(self.path)

    def get_cpdlc_data(self) -> list:
        return self.cpdlc_data

    def get_system_prompt(self) -> str:
        return """You are a CPDLC translation expert. More specifically, you are tasked with extracting ONE CPDLC instruction/message from a natural language based message/instruction from a pilot to an Air Traffic Controller (ATC).
        As general guidelines: The message is in the context of aviation, your outputs should reflect that. For example, flight level X for some number X should follow the convention of FLX, etc.
        If you include a quantity which is associated with an explicit unit in your output, the unit should always be after the quantity.
        Do not add unnecessary complexity to CPDLC messages if it is not stated. For example, do not choose a cpdlc message containing when able unless it is specified or implied to do the instruction when able.
        Do not change the units string contained in the input message unless you are confident that the abreviation you use is accurate. For example, do not change KNOTS to KTS.
        If you output a time, it should follow the 24-hour clock (hh:mm) format.

        Use these relevant CPDLC messages descriptors (each descriptor contains respective fields enclosed by single quotes for CPDLC Message where words surrounded with [] imply an INPUT that MUST be filled in by you if the message is chosen as translation, Intent and Reference Number) as context and choose the appropriate translation based on the intent of the input message:

        {context}

        Translate this pilot/ATC message to CPDLC format and respond ONLY in JSON format where the response should ONLY contain ONE CPDLC translation as follows:
        {{
        "reference": [The correct reference number associated with the corresponding CPDLC Message translation],
        "message": [Translated CPDLC Message],
        "context": [Additional information that the pilot cannot convey through the CPDLC Message that should be given to the ATC. This field should be empty if no additional relevant information is present in the input based on intent.]
        }}
        """
    

# When an ATC uses the application
class ATC(User):
    def __init__(self, cpdlc_data_path=None) -> None:
        super().__init__()
        self.path = CPDLC_UPLINK_PATH if cpdlc_data_path is None else cpdlc_data_path
        self._load_data(self.path)

    def get_cpdlc_data(self) -> list:
        return self.cpdlc_data

    def get_system_prompt(self) -> str:
        return """You are a CPDLC translation expert. More specifically, you are tasked with extracting ONE CPDLC instruction/message from a natural language based message/instruction from an Air Traffic Controller (ATC) to a pilot.
        As general guidelines: The message is in the context of aviation, your outputs should reflect that. For example, flight level X for some number X should follow the convention of FLX, etc.
        If you include a quantity which is associated with an explicit unit in your output, the unit should always be after the quantity.
        Do not add unnecessary complexity to CPDLC messages if it is not stated. For example, do not choose a cpdlc message containing when able unless it is specified or implied to do the instruction when able.
        Do not change the units string contained in the input message unless you are confident that the abreviation you use is accurate. For example, do not change KNOTS to KTS.
        If you output a time, it should follow the 24-hour clock (hh:mm) format.

        Use these relevant CPDLC messages descriptors (each descriptor contains respective fields enclosed by single quotes for CPDLC Message where words surrounded with [] imply an INPUT that MUST be filled in by you if the message is chosen as translation, Intent and Reference Number) as context and choose the appropriate translation based on the intent of the input message:

        {context}

        Translate this pilot/ATC message to CPDLC format and respond ONLY in JSON format where the response should ONLY contain ONE CPDLC translation as follows:
        {{
        "reference": [The correct reference number associated with the corresponding CPDLC message translation],
        "message": [Translated CPDLC Message],
        "context": [Additional information that the air traffic controller cannot convey through the CPDLC Message that should be given to the pilot. This field should be empty if no additional relevant information is present in the input based on intent.]
        }}
        """