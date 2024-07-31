import boto3
import os
import json
import re
import time
from dotenv import load_dotenv

bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")
system_prompt ="Your Task is to correct the given json in the json format provided and return the corrected Json. If any information does not fit the existing categories, return null for those attributes don't create new categories. Ensure no additional notes are included in the end output."
def llama3_8b(prompt ,model, temperature = 0, sys_prompt = system_prompt):
    print(f'model name:{model}')
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
    response = bedrock.invoke_model(
        modelId=payload["modelId"],
        contentType=payload["contentType"],
        accept=payload["accept"],
        body=payload["body"]
    )
 
    # Print the response
    llm_response = json.loads(response['body'].read())
    print(f'checking model{llm_response}')
    # return llm response
    return llm_response['generation']