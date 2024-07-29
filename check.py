from groq import Groq
from dotenv import load_dotenv
from langfuse.decorators import observe
from openai import OpenAI
import os
import json
import boto3
from llm_utils.mistral_llm_call import call_mistral_llm
load_dotenv()
 
# Create a Boto3 client for Bedrock
bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
 
def VLLM_call(input_prompt, api_key, sys_prompt):
    url = os.environ.get('VLLM_URL')
    if not url:
        raise ValueError("VLLM_URL environment variable is not set.")
    client = OpenAI(
        base_url=os.environ.get("VLLM_API_ENDPOINT"),
        api_key=os.environ.get("VLLM_API_SECRET"),
    )
    completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": sys_prompt}, 
        {"role": "user", "content": input_prompt}
    ]
    )
 
    return completion.choices[0].message.content
 
def groq_llm(input_prompt, api_key, model, sys_prompt):
    client = Groq(api_key=api_key)
 
    messages=[
            {
                "role": "system",
                "content": sys_prompt
            },        
            {
                "role": "user",
                "content": input_prompt,
            }
        ]
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model, #"mixtral-8x7b-32768",
        temperature = 0
    )
    return chat_completion.choices[0].message.content
 
def openai_llm(input_prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
 
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]
    )
    return response.choices[0].message.content
 
# Call bedrock LLM model
def aws_bedrock_llm_call(prompt ,model = "meta.llama3-8b-instruct-v1:0", temperature = 0, sys_prompt = "You are a helpful AI assistant"):
 
    # Define Prompt Template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
    {sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
 
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    # Define the payload
    payload = {
        "modelId": model,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "prompt": prompt,
            "temperature": temperature,
            "max_gen_len": 2048,
        })
    }
 
    # call Invoke model
    response = bedrock_client.invoke_model(
        modelId=payload["modelId"],
        contentType=payload["contentType"],
        accept=payload["accept"],
        body=payload["body"]
    )
 
    # Print the response
    llm_response = json.loads(response['body'].read())
    # return llm response
    return llm_response['generation']
 
 
def call_lambda_llm(input_prompt,sys_prompt):
    client = boto3.client('lambda',region_name=os.environ.get("AWS_REGION"))
    payload = {
    "api_key":os.environ.get("LAMBDA_API_KEY"),
    "prompt": input_prompt,
    "temperature": 0.001,
    "max_tokens": 2048,
    "system_prompt":sys_prompt
    }
    try:
        response = client.invoke(
            FunctionName=os.environ.get("LAMBDA_ARN"),
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        response_payload = json.loads(response['Payload'].read())
        response_payload = json.loads(response_payload['body']).get('generated_text').strip()
        return response_payload
    except Exception as e:
        print(f"Error calling Lambda function: {e}")
        return None
 
### uSing Chatgpt as Groq services are down
def generate_chat_completion(input_prompt, model, sys_prompt = "You are a helpful AI assistant which returns in JSON."):
    if model == "vllm":
        api_key = os.environ.get('VLLM_API_KEY')
        return VLLM_call(input_prompt, api_key, sys_prompt)
    elif model=="mistral":
        api_key = os.environ.get('HF_TOKEN')
        return call_mistral_llm(input_prompt, api_key, max_tokens = 1000)
    elif model=="meta.llama3-8b-instruct-v1:0" or model=="meta.llama3-70b-instruct-v1:0":
        return aws_bedrock_llm_call(input_prompt,model=model, sys_prompt=sys_prompt)
    elif model=="lambda":
        return call_lambda_llm(input_prompt,sys_prompt)
    else:
        try:
            api_key = os.environ.get('GROQ_API_KEY')
            groq_response = groq_llm(input_prompt, api_key, model, sys_prompt)
            return groq_response
        except Exception as E1:
            try:
                openai_response = openai_llm(input_prompt)
                return openai_response
            except Exception as E2:
                print(E1, E2)
                return None