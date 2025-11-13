from .config import ALL_CONFIG
import os
import json
import logging
from openai import AzureOpenAI, AsyncAzureOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)

AZURE_ENDPOINT = ALL_CONFIG.get('Urls', {}).get('azure_openai', '')
AZURE_API_KEY = ALL_CONFIG.get('Credentials', {}).get('azure_openai', {}).get('api_key', '')
AZURE_API_VERSION = ALL_CONFIG.get('Credentials', {}).get('azure_openai', {}).get('api_version', '')
AZURE_DEPLOYMENT_ID = ALL_CONFIG.get('Credentials', {}).get('azure_openai', {}).get('model', '')
AZURE_PROXY = ALL_CONFIG.get('Urls', {}).get('proxy', '')


def _init_azure_client():
    """
    Initialize the Azure OpenAI client using environment variables.
    """
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )

def _init_azure_async_client():
    """
    Initialize the Async Azure OpenAI client using environment variables.
    """
    return AsyncAzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )



async def ask_azure_openai(client, max_tokens: int = 200, proxy: str = AZURE_PROXY):
    
    if not hasattr(client, "chat_history"):
        client.chat_history = []

    
    if proxy:
        os.environ["https_proxy"] = proxy

    try:
        azure_client = _init_azure_async_client()

        prompt_text = f"""
        Detect the emotion in the user's input (calm, neutral, happy, angry, fearful, or sad) and respond accordingly without using special characters:

        - Calm/Neutral: Clear, informative, neutral tone.
        - Happy: Positive and enthusiastic.
        - Angry: Acknowledge frustration, offer empathy and solutions.
        - Fearful: Offer reassurance and comfort.
        - Sad: Show empathy and offer a kind, uplifting response.

        Now, respond to the following input with the appropriate emotional tone: {str(client.user_input_txt)}
        """

       
        client.chat_history.append({"role": "user", "content": prompt_text})

        response = await azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=client.chat_history,
            max_tokens=max_tokens
        )

        print(response)
        message = response.choices[0].message.content.strip()

      
        client.chat_history.append({"role": "assistant", "content": message})

        return message
    
    except Exception as e:
        logger.error("Error in Azure Openai pipeline: {}".format(e))
        raise

    finally:
        if proxy:
            os.environ.pop("https_proxy", None)


def classify_emotion(text: str, proxy: str = AZURE_PROXY):
    """
    Classify emotion in the given text using Azure GPT.
    """
    
    if proxy:
        os.environ["https_proxy"] = proxy

    try:
        azure_client = _init_azure_client()

        prompt = (
            f"Classify the emotion in the text: [{text}]. "
            "Output only one word from [calm, happy, angry, fearful, sad, neutral]. "
            "If uncertain, return 'neutral'."
        )

        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        emotion = response.choices[0].message.content.strip().lower()
        if emotion == "cal":
            emotion = "calm"
        if emotion == "fear":
            emotion = "fearful"

        return emotion
    except Exception as e:
        logger.error("Error while Emotion detection in Azure Openai pipeline: {}".format(e))
        raise
        
    finally:
        if proxy:
            os.environ.pop("https_proxy", None)
            
            
async def chatgpt_entity_extractor_insurance(client, user_input="",proxy: str = AZURE_PROXY):
    
    if proxy:
        os.environ["https_proxy"] = proxy

    try:
        azure_client = _init_azure_async_client()

        prompt_text = f"""

        Now, respond to the following input with the appropriate emotional tone: Additionally, extract the following entities if present:  
 
        - **credit_card_last_4:** XXXX  
        - **zip_code:** XXXXX 
        - **policy_number:** XX
        - **billing_case_number:** XXXXXX

        If an entity is missing, indicate it as **"Not provided."**  

        Now, analyze the following input, extract the entities:""" + str(user_input)
            
       
        headers = {
        'Content-Type': 'application/json'
        }

    
        
        response = await azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
            
                
        patterns = {
                    "credit_card_last_4": r"\*\*credit_card_last_4\:\*\* (\d+)",  # Match any number of digits
                    "zipcode": r"\*\*zipcode\:\*\* (\d+)",  # Match any number of digits
                    "policy_number": r"\*\*policy_number\:\*\* (\d+)",
                    "billing_case_number": r"\*\*billing_case_number\:\*\* (\d+)"}

        for key, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                client.extracted_entity_dict[key] = match.group(1)

        
        for entity_type, entity_value in client.extracted_entity_dict.items():
            if re.search(r'\b' + re.escape(entity_value) + r'\b', output):
                spaced_value = ' '.join(entity_value)
                output = re.sub(r'\b' + re.escape(entity_value) + r'\b', spaced_value, output)
            elif re.search(r'\b' + re.escape(entity_value) + r'(?=\W)', output):
                spaced_value = ' '.join(entity_value)
                output = re.sub(r'\b' + re.escape(entity_value) + r'(?=\W)', spaced_value, output)
        
                
        return output

    except Exception as e:
        logger.error("Error in extracting entities via azure openai: {}".format(e))
        return user_input
    
    finally:
        if proxy:
            os.environ.pop("https_proxy", None)
     
    
            
