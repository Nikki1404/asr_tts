import aiohttp
import json
import logging
import re
import requests
import time
import uuid
from typing import Any, Callable, Dict, Optional

from google.cloud.dialogflowcx_v3.services.agents.client import AgentsClient
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.types import session
from .azure_openai_prompt import ask_azure_openai, chatgpt_entity_extractor_insurance
from .config import ALL_CONFIG


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ALL_CONFIG["PATH"]["log_file_global"])
logger.addHandler(file_handler)


# -----------------------------
# Helper utilities
# -----------------------------
def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0, verify: Optional[bool] = None) -> Optional[requests.Response]:
    """Perform an HTTP POST with JSON payload and return the response."""
    headers = {"Content-Type": "application/json"}
    try:
        if verify is None:
            return requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout)
        return requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout, verify=verify)
    except Exception as exc: 
        logger.error(f"POST {url} failed: {exc}")
        return None


def _set_session_if_empty(client: Any, session_id: Optional[str]) -> None:
    """Set client's session_id if currently empty."""
    if getattr(client, "session_id", "") == "" and session_id:
        client.session_id = session_id
        print(f"session id updated : {client.session_id}")
        print("*" * 100)


def separate_digits_with_space(input_string):
    separated_string = re.sub(r'(\d)', r'\1 ', input_string)
    return separated_string.strip()  



def replace_numeric_entities_by_name(json_body, entity_types, client):
    agent_states = json_body.get("Agent_states", [])
    output = json_body.get("response", "")
    
    for state in agent_states:
        dynamic_planner = state.get("Dynamic_Planner", {})
        
        if isinstance(dynamic_planner, dict):
            entities = dynamic_planner.get("extracted_entity", {})
            
            for entity_type in entity_types:
                if entity_type in entities and entities[entity_type] != "":
                    client.extracted_entity_dict[entity_type] = entities[entity_type]
    
    # Modify the output value with white space if entity found
    for entity_type, entity_value in client.extracted_entity_dict.items():
        entity_value = str(entity_value)
        if re.search(r'\b' + re.escape(entity_value) + r'\b', output):
            spaced_value = ' '.join(entity_value)
            output = re.sub(r'\b' + re.escape(entity_value) + r'\b', spaced_value, output)
        elif re.search(r'\b' + re.escape(entity_value) + r'(?=\W)', output):
            spaced_value = ' '.join(entity_value)
            output = re.sub(r'\b' + re.escape(entity_value) + r'(?=\W)', spaced_value, output)
    
    return output
     
    
async def smart_agent_dialogue_manager(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    
    result = None
    
        
    try:
        url = ALL_CONFIG["Urls"]["autonomous_agents"]['healthcare-agent']
        
        payload = json.dumps({
        "user_input": user_input,
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)
            
        if response.status_code == 200:
            
            response_data = response.json()
            
            entity_types = ["account_number_last_4","phone_number","zip_code","service_account_number_last_4"]
            client.tts_response = replace_numeric_entities_by_name(response_data, entity_types, client)
            
            
            result = response_data.get("response")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"smart agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in SmartAgent pipeline: {}".format(e))

        return None


async def healthcare_demo_faq(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        url = ALL_CONFIG["Urls"]["training_ai"]["healthcare_demo"]["faq"]
        

        if client.session_id == "":
            client.session_id = str(uuid.uuid4())

        payload = json.dumps({
            "user_input": user_input,
            "session_id": client.session_id,
        })

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer')
            session_id = response_data.get('session_id')

            if customer_reply is None:
                logger.error(f"Unexpected response format from healthcare_demo_faq api: {response_data}")
                return "Error: Could not understand healthcare_demo_faq response."

            
            print("customer_reply: ", customer_reply)
            
            
            return {"type": "server_transcript", "text":customer_reply, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling healthcare_demo_faq api: {e}")

        return None

async def healthcare_mock_call(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        url = ALL_CONFIG["Urls"]["training_ai"]["healthcare_demo"]["mock"]
        

        payload = json.dumps({
            "user_input": user_input,
            "persona": "Neutral",
            "session_id": client.session_id,
            "scenario": "**Situation Summary:**\nThe discussion revolves around a prior authorization request submitted by Dr. Emily Carter for her patient, John Doe, who is a 58-year-old male diagnosed with Type 2 Diabetes Mellitus. John has been experiencing breathlessness, wheezing, and postprandial daytime sleepiness. His current medications include Kazano for diabetes, Losartan for hypertension, and Lipitor for cholesterol management. Due to his condition, Dr. Carter plans to initiate Lantus, a long-acting insulin. The call aims to ensure the authorization process is smooth and to address any potential issues or additional information required by the insurance provider.\n\n**Doctor's Call Opening:**\n'Hello, this is Dr. Emily Carter from Springfield Endocrinology Clinic. I am calling regarding my patient, John Doe, for whom I have submitted a prior authorization request for Lantus insulin therapy. I would like to discuss the status of this request and provide any additional information needed to expedite the process.'\n\n**Satisfaction Trigger:**\nThe doctor will be satisfied if the prior authorization is approved without any complications or if the agent provides clear guidance on any additional steps required.\n\n**Dissatisfaction Trigger:**\nThe doctor will be dissatisfied if there are unnecessary delays, lack of clarity, or if additional information is requested without a clear explanation.\n\n**Ending the Call:**\n- **If Satisfied:** 'Thank you for your assistance. I appreciate your help in ensuring my patient receives the necessary medication promptly. Have a great day!'\n- **If Dissatisfied:** 'I am concerned about the delays and lack of clarity in this process. Can you please escalate this matter or provide a more detailed explanation of what is needed? Thank you.'",
            "customer_details": {
            "patient_info": {
                "Name": "John Doe",
                "Age/Gender": "58 / Male",
                "Member ID": "X8J29L7345",
                "Indication": "Type 2 Diabetes Mellitus",
                "Chief Complaint": "Breathlessness, wheezing, and postprandial daytime sleepiness"
            },
            "Diagnosis": {
                "Primary": "Type 2 Diabetes Mellitus (ICD-10: E11.65 – Type 2 diabetes mellitus with hyperglycemia)",
                "Secondary": [
                "Essential Hypertension (ICD-10: I10)",
                "Raised LDL Cholesterol (ICD-10: E78.00)"
                ]
            },
            "Current Medications": [
                "Kazano (Alogliptin + Metformin) for Diabetes",
                "Losartan for Hypertension",
                "Lipitor (Atorvastatin) for Cholesterol Management"
            ],
            "Plan": "Initiate Lantus (Long-acting Insulin)",
            "Provider": "Dr. Emily Carter (NPI: B2M7P6349)",
            "Clinic Name": "Springfield Endocrinology Clinic",
            "Clinic Contact": "(555) 678-9012"
            },
            "n_question": "100",
            "closing_turn": "2",
            "utils": {}
        }
        )

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer').get('customer_query')
            call_status = response_data.get('answer').get('call_status')
            session_id = response_data.get('session_id')

            if client.session_id == "":
                client.session_id = session_id
            if customer_reply is None:
                logger.error(f"Unexpected response format from mock call api: {response_data}")
                return "Error: Could not understand mock call response."
            
                
                print("customer_reply: ", customer_reply)
                
            return {"type": "server_transcript", "text":customer_reply, "call_status":call_status, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling healthcare_mock_call api: {e}")

        return None
    
def healthcare_highmark_auto_script(client) -> None:

        url = ALL_CONFIG["Urls"]["training_ai"]["highmark"]["autoscript"]

        payload_dict = {
        "domain": "healthcare",
        "demo_name": "highmark_call_transfer_successful",
        "text_document": "Agent: Hi, good morning David. My name is Angel, and I’m calling from ABC Blue Cross Blue Shield. This call is recorded for quality and training. How have you been feeling lately? Customer: Oh… hi. Well, I’m still a bit sore, honestly. Some days are better than others. Agent: I’m sorry to hear that, I know recovery can be up and down. It’s good you’re noticing those better days, though. The reason I’m calling today is to let you know that, as part of your plan, you have access to a personal, dedicated registered nurse — at no extra cost. This nurse can answer your health questions, help you find services, support your recovery goals, and even help with scheduling doctor’s appointments. I’d love to connect you with them today so they can see how best to support you. Customer: That sounds… reassuring. Will she do some therapy too? Agent: That’s a great question, David. The nurse won’t provide therapy directly, but our Case Manager can walk you through your therapy options and guide you in getting it arranged. Customer: I see… I just… I feel like I’m falling behind in my recovery. I need a nurse to come here, and I need some physical therapy. Agent: I hear you, and you’re not alone in feeling that way. Many people feel that same concern during this stage of recovery. The Case Manager can talk about arranging in-home nurse visits and help you set up physical therapy. You don’t need to write anything down — they’ll make sure you have clear next steps. Should I connect you now, or would you prefer a time later today? Customer: No, it’s fine. I’m free now. Agent: Perfect. Before I transfer you, to protect your health privacy, I just need to confirm either your date of birth or your complete mailing address. Customer: My address is 04:13 Purple Drive, Happy Town, Pennsylvania. One five, oh, six eight. Agent: Thank you for confirming, David. I’ll let the Case Manager know you’re available right now and place you on hold for about two minutes — does that work for you? Customer: Yes, that’s fine. Thank you. Agent: You’re welcome. Before I do that, is this the best number to reach you if we get disconnected? Customer: Yes — it’s 555 123. 4364. Agent: Thank you — 555 123. 4364, correct? Customer: Yes, that’s right. Agent: Great. I’ll connect you now. Please hold for a moment. Agent: David, thank you for holding. Your Case Manager is now on the line, and I’ve already shared your details, so you don’t have to repeat yourself. If the call drops, she’ll call you back on your preferred number. I hope this conversation gives you the clarity and support you’ve been looking for. Customer: I really appreciate that. Thank you for checking in on me. Agent: You’re very welcome, David. Wishing you comfort and steady progress in your recovery.",
        "customer_details": {
            "customer_data": {
            "customer_personal_info": {
                "Name": "David Martinez",
                "DoB": "15-Jun-1958",
                "Age": "67",
                "Email ID": "David.Martinez@email.com",
                "Phone Number": "(555) 321-7890",
                "Address": "413 Purple Drive, Happy Town, PA, 15068"
            },
            "customer_order_info": [
                {
                "Date": "7/15/2025",
                "Time": "11:00 AM (estimated)",
                "Agent Name": "Princess",
                "Reason for Call": "Inform about personal dedicated nurse via Highmark BCBS",
                "Outcome": "Verified DOB, nurse connection initiated",
                "Call Duration": "Approx. 8 minutes"
                }
            ],
            "title": "Assistance with Access to Personal Dedicated Nurse for Walter"
            }
        },
        "persona": "Emotional",
        "input_type": "call_transcript",
        "sample_example": {
            "sample_document_input_example": "",
            "sample_customer_output_example": ""
        }
        }
        headers = {
        'Content-Type': 'application/json'
        }
        
        if getattr(client, "nlp_engine_config", None):
            for key in payload_dict:
                if key in client.nlp_engine_config:
                    payload_dict[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(payload_dict)
        

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
        if response and response.status_code == 200:
            client.auth_config = response.json()

        

async def healthcare_highmark_mock_call(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        
        if not client.auth_config:
            healthcare_highmark_auto_script(client)
            
            
        
        url = ALL_CONFIG["Urls"]["training_ai"]["highmark"]["mock"]
        client.auth_config.update({
            "utils": {},
            "session_id": client.session_id,
            "user_input": user_input,
            "history": "",
            "history_list": [],
            "domain": "healthcare",
            "demo_name": "highmark_call_transfer_successful",
            "n_question": "100",
            "closing_turn": "2"
        })
        
        if getattr(client, "nlp_engine_config", None):
            for key in client.auth_config:
                if key in client.nlp_engine_config:
                    client.auth_config[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(client.auth_config)

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer').get('customer_query')
            call_status = response_data.get('answer').get('call_status')
            session_id = response_data.get('session_id')

            if client.session_id == "":
                client.session_id = session_id
            if customer_reply is None:
                logger.error(f"Unexpected response format from mock call api: {response_data}")
                return "Error: Could not understand mock call response."
            
            cleaned_customer_reply = re.sub(r'this raksha bandhan gift', '', customer_reply, flags=re.IGNORECASE)
            
            return {"type": "server_transcript", "text":cleaned_customer_reply, "call_status":call_status, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling healthcare_mock_call api: {e}")

        return None
    
def retail_next_auto_script(client) -> None:

        url = ALL_CONFIG["Urls"]["training_ai"]["retail_next"]["autoscript"]

        payload_dict = {
        "domain": "Retail",
        "demo_name": "Next",
        "text_document": "CALLER: Hello. AGENT: Thank you for calling ABC. This is Yasin. How can I help you today? CALLER: I ordered the wrong cushion and meant to refuse the delivery, but it just arrived. Can my friend return it to your drop-off point? AGENT: I see. Yes, your friend can return the item. They’ll just need the order confirmation or any proof of purchase. CALLER: Okay, I’ll send her a photo of the email—I’m 73 and not great with tech. AGENT: That’s absolutely fine. A photo will work, or she can log into your account to show the order. As an alternate I can arrange home pickup directly from your doorstep CALLER: Thanks but I have already spoken to my friend. For now, my friend will return it. Thanks AGENT:  Sure. sir CALLER: Thank you AGENT: Have a nice day CALLER: Good bye",
        "customer_details": {
            "customer_personal_info": {
            "Name": "Jude Tarrant",
            "DoB": "19-02-68",
            "Age": "73",
            "Email ID": "Jude1952.mt@gmail.com",
            "Phone Number": "",
            "Customer ID": "AB 574751",
            "Address Line 1": "",
            "Address Line 2": "",
            "Post code": "B77. 3AX."
            },
            "customer_order_info": [
            {
                "Item ID": "IT76543",
                "Description": "Brown Cushion",
                "Qty": 1,
                "Amount": "£5.0",
                "Purchase Date": "3/22/2025",
                "Status": "Delivered"
            },
            {
                "Item ID": "IT76544",
                "Description": "pink blue trouser",
                "Qty": 6,
                "Amount": "£34.0",
                "Purchase Date": "1/12/2025",
                "Status": "Delivered"
            },
            {
                "Item ID": "IT76545",
                "Description": "Red Italian Rug",
                "Qty": 1,
                "Amount": "£75.0",
                "Purchase Date": "12/24/2024",
                "Status": "Return Successful"
            },
            {
                "Item ID": "IT76546",
                "Description": "Crystal Flower Vase",
                "Qty": 1,
                "Amount": "£3.5",
                "Purchase Date": "8/6/2024",
                "Status": "Delivered"
            }
            ]
        },
        "persona": "",
        "input_type": "",
        "sample_example": {
            "sample_document_input_example": "",
            "sample_customer_output_example": ""
        }
        }
        
        if getattr(client, "nlp_engine_config", None):
            for key in payload_dict:
                if key in client.nlp_engine_config:
                    payload_dict[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(payload_dict)
        
        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
        if response and response.status_code == 200:
            client.auth_config = response.json()
            

async def retail_next_mock_call(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        
        if not client.auth_config:
            retail_next_auto_script(client)
            
            
        url = ALL_CONFIG["Urls"]["training_ai"]["retail_next"]["mock"]
        client.auth_config.update({
            "utils": {},
            "session_id": client.session_id,
            "user_input": user_input,
            "history": "",
            "history_list": [],
            "domain":  "Retail",
            "demo_name": "Next",
            "n_question": "100",
            "closing_turn": "2"
        })
        
        if getattr(client, "nlp_engine_config", None):
            for key in client.auth_config:
                if key in client.nlp_engine_config:
                    client.auth_config[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(client.auth_config)

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer').get('customer_query')
            call_status = response_data.get('answer').get('call_status')
            session_id = response_data.get('session_id')

            if client.session_id == "":
                client.session_id = session_id
            if customer_reply is None:
                logger.error(f"Unexpected response format from mock call api: {response_data}")
                return "Error: Could not understand mock call response."
            
            cleaned_customer_reply = customer_reply
            return {"type": "server_transcript", "text": cleaned_customer_reply, "call_status": call_status, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error calling retail_mock_call api: {e}")

        return None
    
def banking_inspira_auto_script(client) -> None:

        url = ALL_CONFIG["Urls"]["training_ai"]["banking_inspira"]["autoscript"]

        payload_dict = {
            "domain": "banking",
            "demo_name": "inspira",
            "text_document": "Agent: Thank you for calling Spider financial who we have the pleasure of speaking with.\nCustomer: Hi, my name is Austin A U S T I N and my last name is Ut U T L E Y.\nAgent: Okay, and how can I help you today?\nCustomer: I got one of my old job rolled over my 401 K to you guys and I was trying to access my account but it will not send me a text to verify my number. It keeps on telling me I have to call this number. So I was wondering if you guys could help me verify my number so I can access my new account with you guys.\nAgent: Okay. Let me try to find the account with your information. May I have the last four digits of your account number for security verification?\nCustomer: Yes, the last four is 2904.\nAgent: 2904. Okay, let us see. May I have your date of birth?\nCustomer: Yes, it is 01-18-1990.\nAgent: May I have your full address?\nCustomer: Yes, my full address is Po Box 99, 4-Stanford, Illinois 61774.\nAgent: Got it. And do you remember the name of your former employer?\nCustomer: Yes, it is JJ X Enterprises Incorporated.\nAgent: Okay, got it. And finally, what is the best phone number to reach you?\nCustomer: It is 309-665-2834.\nAgent: Okay, let me confirm back, 309-665-2834, right?\nCustomer: Yes.\nAgent: Thank you so much. I can see that you have a rollover traditional IRA account. You mentioned that you are having issues receiving the code on the website.\nCustomer: Yeah, I was trying to verify my number and it never text me the verification code. But then when I tried doing it again, it told me to call this number.\nAgent: Okay, let us see. And do you already know what you want to do with the account?\nCustomer: I was thinking if I could maybe send it to a bank account.\nAgent: Withdraw. Okay.\nCustomer: Yes.\nAgent: Okay, let us see. Because I can see that your login was created. So there is an email registered. Did you try maybe to sign out and log in one more time?\nCustomer: Okay, yep, I will try that right now.\nAgent: Okay. Do you need the link to the website or do you already have it?\nCustomer: I have got an email from it saying to confirm my email.\nAgent: Okay.\nCustomer: Okay. Now it is telling me to type in my number again, so I am going to try that. See if this works now. Yeah. It said we were unable to send the code. Please verify the phone number provided and try again or call client services at this number. Do you have the actual website?\nAgent: Yes, it seems like you are in the right place. But let me reset the alert here in order that you can receive the code. Give me one second.\nCustomer: Okay.\nAgent: Okay, I already did it, so maybe try one more time to resend the code.\nCustomer: Okay, I will go to confirm my account. Log in. Now it is telling me to make a new account. Is that what I am supposed to do?\nAgent: No. Maybe try just to log in one more time.\nCustomer: Okay, I will try that.\nAgent: Let me know if you could receive the code.\nCustomer: Okay. Alright, now I am going to type in my number again real quick.\nAgent: Okay.\nCustomer: Yeah, it keeps on telling me the same thing. I tried logging in and it took me back to the number screen.\nAgent: Okay, no worries. Let us try with another link. I am going to send you a one-time link so you can request for the distribution there. What is your email address?\nCustomer: My email address is austin.utley44@example.com.\nAgent: Okay, got it. I already sent to you the email. Please verify if you receive it.\nCustomer: Alright. Yep, I got it.\nAgent: Perfect. So you can request the distribution. Once you finish that request, the process takes around five business days. Also, remember that you have a closing fee of $25 and penalties and taxes may apply. You will receive the 1099 form next year.\nCustomer: Okay.\nAgent: Do you need me to stay on the line with you to help you with that link, or would you rather do it by yourself?\nCustomer: Let me see.\nAgent: Okay.\nCustomer: I think I can get this, so I think that is all I would need.\nAgent: Okay, perfect. Do you have any other questions for me?\nCustomer: Nope, that is it.\nAgent: Thank you so much for calling Spider Financial and have a great day. Take care.\nCustomer: Thank you. You too.\nAgent: Thank you. Bye bye. Bye bye.",
            "customer_details": {
                "customer_data": {
                "customer_personal_info": {
                    "Name": "Austin Utley",
                    "DoB": "01-18-1990",
                    "Age": "33",
                    "Email ID": "austin.utley44@example.com",
                    "Phone Number": "309-665-2834",
                    "Address": "Po Box 99, 4-Stanford, Illinois 61774"
                },
                "customer_order_info": [],
                "title": "Rollover 401k Account Verification Issue Call Transcript"
                }
            },
            "persona": "Neutral",
            "input_type": "call_transcript",
            "sample_example": {
                "sample_document_input_example": "",
                "sample_customer_output_example": ""
            }
            }
        
        if getattr(client, "nlp_engine_config", None):
            for key in payload_dict:
                if key in client.nlp_engine_config:
                    payload_dict[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(payload_dict)
        
        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
        
        
        if response and response.status_code == 200:
            client.auth_config = response.json()
            
            
async def banking_inspira_mock_call(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        
        if not client.auth_config:
            banking_inspira_auto_script(client)
            
            
        url = ALL_CONFIG["Urls"]["training_ai"]["banking_inspira"]["mock"]
        client.auth_config.update({
            "utils": {},
            "session_id": client.session_id,
            "user_input": user_input,
            "history": "",
            "history_list": [],
            "domain":  "banking",
            "demo_name": "inspira",
            "n_question": "100",
            "closing_turn": "2"
        })
        
        if getattr(client, "nlp_engine_config", None):
            for key in client.auth_config:
                if key in client.nlp_engine_config:
                    client.auth_config[key] = client.nlp_engine_config[key]
        
        payload = json.dumps(client.auth_config)
    

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10, verify=False)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer').get('customer_query')
            call_status = response_data.get('answer').get('call_status')
            session_id = response_data.get('session_id')

            if client.session_id == "":
                client.session_id = session_id
            if customer_reply is None:
                logger.error(f"Unexpected response format from mock call api: {response_data}")
                return "Error: Could not understand mock call response."
            
            
            return {"type": "server_transcript", "text":customer_reply, "call_status":call_status, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling retail_mock_call api: {e}")

        return None

async def utility_faq(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        url = ALL_CONFIG["Urls"]["training_ai"]["utility"]["faq"]
        

        if client.session_id == "":
            client.session_id = str(uuid.uuid4())

        payload = json.dumps({
            "utils": {},
            "session_id": client.session_id,
            "user_input": user_input
        })

        headers = {
        'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer')
            session_id = response_data.get('session_id')

            if customer_reply is None:
                logger.error(f"Unexpected response format from healthcare_demo_faq api: {response_data}")
                return "Error: Could not understand healthcare_demo_faq response."

            
            print("customer_reply: ", customer_reply)
            
            
            return {"type": "server_transcript", "text":customer_reply, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling healthcare_demo_faq api: {e}")

        return None

async def utility_mock_call(client):
    user_input = str(getattr(client, "user_input_txt", ""))
    result = None
        
    try:
        url = ALL_CONFIG["Urls"]["training_ai"]["utility"]["mock"]
        

        if client.session_id == "":
            client.session_id = str(uuid.uuid4())

        payload = json.dumps({
            "scenario": "**Situation Summary:**\n\nJohn Doe, residing at 82 Portland Road, has returned from holiday to find his Vaillant boiler displaying an F75 error code, indicating a pump water shortage. He has no heating or hot water and only a small electric radiator as an alternative heating source. John has an 80-year-old vulnerable person in the house with angina and asthma, making it critical to restore heating immediately. John has tried resetting the boiler without success and needs urgent assistance. The agent offers an appointment for the next day but escalates the case to a same-day emergency due to the vulnerable person in the household. John is informed of a £60 excess fee on his Homecare 4 policy, which he agrees to pre-authorize.\n\n**Call Opening Response:**\n\n'Hello, yes, we've just returned from holiday, and our boiler isn't working. It's a Vaillant, and it’s showing an F75 fault—pump water shortage.'\n\n**Satisfaction Trigger:**\n\nJohn should feel satisfied if the agent successfully escalates the case for same-day emergency service and confirms the engineer's visit for today. He should end the call with a happy greeting: 'Brilliant, thank you so much for your help.'\n\n**Dissatisfaction Trigger:**\n\nJohn should feel dissatisfied if the agent fails to prioritize the case for same-day service or if there are issues with the pre-authorization process for the excess fee. He should respond neutrally: 'I appreciate your help, but I'm concerned about the delay. Can you please ensure this is treated as an emergency?'",
            "customer_details": {
                "customer_data": {
                    "name": "John Doe",
                    "address": "82 Portland Road",
                    "boiler_make": "Vaillant",
                    "boiler_error_code": "F75",
                    "vulnerable_person": {
                        "age": 80,
                        "conditions": [
                            "angina",
                            "asthma"
                        ]
                    },
                    "alternative_heating": [
                        "small electric radiator"
                    ],
                    "contact_number_last_digits": "849",
                    "policy": "Homecare 4"
                },
                "title": "Emergency Boiler Repair Request"
            },
            "persona": "Neutral",
            "session_id": client.session_id,
            "user_input": user_input,
            "n_question": "100",
            "closing_turn": "2",
            "domain": "Gas"
        })

        headers = {
            'Content-Type': 'application/json'
        }

        response = _http_post_json(url, json.loads(payload), timeout=10)
            
        if response and response.status_code == 200:
            response_data = response.json()
            customer_reply = response_data.get('answer').get('customer_query')
            call_status = response_data.get('answer').get('call_status')
            session_id = response_data.get('session_id')

            if customer_reply is None:
                logger.error(f"Unexpected response format from healthcare_demo_faq api: {response_data}")
                return "Error: Could not understand healthcare_demo_faq response."

            
            print("customer_reply: ", customer_reply)
            
            
            return {"type": "server_transcript", "text":customer_reply, "call_status":call_status, "session_id":session_id}
    except Exception as e:
        logger.error(f"Error calling healthcare_demo_faq api: {e}")

        return None

async def dialogflow_mock_call(client):
    
    result = None
        
    try:
        url = ALL_CONFIG["Urls"]["dialogflow"]["banking_demo"]
        
        payload = json.dumps({
            "text": str(getattr(client, "user_input_txt", "")),
            "session_id": client.session_id
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)
            
        if response.status_code == 200:
            
            response_data = response.json()
            
            
            result = response_data.get("message")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            return result
        else:
            print ({"error": f"smart agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in dialogflow_mock_call: {}".format(e))

        return None


async def fnol_agent(user_input = "", session_id = ""):
    print("inside insurance dialogue mangent ------------------")
    project_id = "eci-ugi-digital-ccaipoc"

    location_id = "global"

    agent_id = "8e79c24e-7a85-40bc-b423-173c5c220b2e"
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    language_code = "en-us"

    session_path = f"{agent}/sessions/{session_id}"
    

    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        
        client_options = {"api_endpoint": api_endpoint}
    

    session_client = SessionsClient(client_options=client_options)

    text_input = session.TextInput(text=user_input) 
    query_input = session.QueryInput(text=text_input, language_code=language_code)
    request = session.DetectIntentRequest(
        session=session_path, query_input=query_input 
    )
    response = session_client.detect_intent(request=request)

    print(f"response found from insurance agent:----------{response}")

    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    

    agent_response = ' '.join(response_messages)

    return agent_response


async def pharma_agent(user_input = "", session_id = ""):
    project_id = "eci-ugi-digital-ccaipoc"

    location_id = "global"

    agent_id = "1c89195c-4e69-4054-a8cd-9d823acb3f7e"
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    language_code = "en-us"

    session_path = f"{agent}/sessions/{session_id}"
    

    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        
        client_options = {"api_endpoint": api_endpoint}
    
    session_client = SessionsClient( client_options=client_options)

    text_input = session.TextInput(text=user_input) 
    query_input = session.QueryInput(text=text_input, language_code=language_code)
    request = session.DetectIntentRequest(
        session=session_path, query_input=query_input 
    )
    response = session_client.detect_intent(request=request)

    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    

    agent_response = ' '.join(response_messages)

    return agent_response


async def healthcare_address_change_agent(user_input = "", session_id = ""):
    project_id = "eci-ugi-digital-ccaipoc"

    location_id = "global"

    agent_id = "88969b57-e560-4a58-b439-aad324b7dbef"
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    language_code = "en-us"

    session_path = f"{agent}/sessions/{session_id}"
    

    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        
        client_options = {"api_endpoint": api_endpoint}
    
    session_client = SessionsClient(client_options=client_options)

    text_input = session.TextInput(text=user_input) 
    query_input = session.QueryInput(text=text_input, language_code=language_code)
    request = session.DetectIntentRequest(
        session=session_path, query_input=query_input 
    )
    response = session_client.detect_intent(request=request)

    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    

    agent_response = ' '.join(response_messages)

    return agent_response



async def insurance_agent_dialogue_manager(client):
    
    result = None
    
        
    try:
        url = ALL_CONFIG["Urls"]["autonomous_agents"]['insurance-agent']
        

        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)
            
        if response.status_code == 200:
            
            response_data = response.json()
            result = response_data.get("response")
            
            if result and any(char.isdigit() for char in result):
                client.tts_response = await chatgpt_entity_extractor_insurance(client = client, user_input=result)
                logger.info(f"client.tts response is: {client.tts_response}")
            
            
            
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"smart agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in insurance_agent_dialogue_manager: {}".format(e))

        return None
    
async def insurance_pnc(client):
    
    result = None
    
    

    print("coming for insurance_pnc...")
    try:
        url = ALL_CONFIG["Urls"]["autonomous_agents"]["insurance-pnc-agent"]

        

        print("user_input: ", str(getattr(client, "user_input_txt", "")))

        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)
            
        if response.status_code == 200:
            
            response_data = response.json()
            result = response_data.get("response")

            
            print("result: ", result)
            
            if result and any(char.isdigit() for char in result):
                client.tts_response = chatgpt_entity_extractor_insurance(client = client, user_input=result)
                logger.info(f"client.tts response is: {client.tts_response}")
            
            
            
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"smart agent Request failed with status code {response.status_code}"})
    except Exception as e:
        logger.error("Error in insurance_agent_pnc_dialogue_manager: {}".format(e))

        return None

async def banking_agent(client):
    result = None
    
    try:
        url = ALL_CONFIG["Urls"]["autonomous_agents"]['banking-agent']
        

        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)
            
        if response.status_code == 200:
            
            response_data = response.json()
            
            
            entity_types = ["account_number_last_4","phone_number","zip_code","service_account_number_last_4"]
            client.tts_response = replace_numeric_entities_by_name(response_data, entity_types, client)
            
            result = response_data.get("response")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"smart agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in banking_agent: {}".format(e))

        return None

async def utility_agent(client):
    result = None

    try:
        url = ALL_CONFIG["Urls"]["autonomous_agents"]['utility-agent']


        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload,timeout=10)

        if response.status_code == 200:

            response_data = response.json()
            
            entity_types = ["account_number_last_4","phone_number","zip_code","service_account_number_last_4"]
            client.tts_response = replace_numeric_entities_by_name(response_data, entity_types, client)
            
            result = response_data.get("response")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)

            
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"utility agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in utility_agent: {}".format(e))

        return None


async def banking_os_agent(client):
    result = None
    
    try:
        
        url = ALL_CONFIG["Urls"]["autonomous_agents"]["banking-os-agent"]
        

        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=6.0)
            
        if response.status_code == 200:
            
            response_data = response.json()
            
            
            result = response_data.get("response")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"banking-os-agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in banking-os-agent: {}".format(e))

        return None

async def healthcare_preauth_agent(client):
    result = None
    
    try:
        
        url = ALL_CONFIG["Urls"]["autonomous_agents"]["healthcare-preauth-agent"]

        payload = json.dumps({
        "user_input": str(getattr(client, "user_input_txt", "")),
        "session_id": client.session_id,
        "utils": {}
        })
        headers = {
        'Content-Type': 'application/json'
        }

        
        response = requests.request("POST", url, headers=headers, data=payload,timeout=6.0)
            
        if response.status_code == 200:
            
            response_data = response.json()
            
            
            result = response_data.get("response")
            if client.session_id =="":
                client.session_id= response_data.get("session_id","")
                print(f"session id updated : {client.session_id}")
                print("*"*100)
                
            session_id = response_data.get("session_id")
            agent_states = response_data.get("Agent_states")
            return {"type":"server_transcript", "text": result, "session_id": session_id, "agent_states": agent_states}
        else:
            print ({"error": f"healthcare-preauth-agent Request failed with status code {response.status_code}"})

    except Exception as e:
        logger.error("Error in healthcare-preauth-agent: {}".format(e))

        return None

async def dialogue_manager(client):
    fallback = "Sorry, It's not you. It's me! Please try again after sometime."
    fallback_empty = "Sorry, I couldn't understand you. Please try again."
    
    result = ""
    agent = client.nlp_engine
    
    ROUTES: Dict[str, Callable[[Any], Any]] = {
        "healthcare-agent": lambda c: smart_agent_dialogue_manager(c),
        "insurance-agent": lambda c: insurance_agent_dialogue_manager(c),
        "banking-agent": lambda c: banking_agent(c),
        "utility-agent": lambda c: utility_agent(c),
        "banking-os-agent": lambda c: banking_os_agent(c),
        "healthcare-preauth-agent": lambda c: healthcare_preauth_agent(c),
        "insurance-pnc": lambda c: insurance_pnc(c),
        "chatgpt": lambda c: ask_azure_openai(client=c),
        "banking-customer": lambda c: mock_call(c),
        "fnol-agent": lambda c: fnol_agent(user_input=str(getattr(c, "user_input_txt", "")), session_id=c.client_id),
        "pharma-agent": lambda c: pharma_agent(user_input=str(getattr(c, "user_input_txt", "")), session_id=c.client_id),
        "healthcare-address-change-agent": lambda c: healthcare_address_change_agent(user_input=str(getattr(c, "user_input_txt", "")), session_id=c.client_id),
        "healthcare-demo": lambda c: healthcare_mock_call(c),
        "healthcare-demo-faq": lambda c: healthcare_demo_faq(c),
        "utility-faq": lambda c: utility_faq(c),
        "utility-mock-call": lambda c: utility_mock_call(c),
        "healthcare-highmark-mock-call": lambda c: healthcare_highmark_mock_call(c),
        "retail-next-mock-call": lambda c: retail_next_mock_call(c),
        "banking-inspira-mock-call": lambda c: banking_inspira_mock_call(c),
    }

    handler = ROUTES.get(agent)
    if handler:
        result = await handler(client)
 
 
 
    print(f"result found :{result}")
    if result in [""]:
        result = fallback_empty
    if result  is None:
        result = fallback
                
    return result


