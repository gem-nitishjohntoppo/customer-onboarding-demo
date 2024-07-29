import boto3
import json
import os
import botocore.exceptions
from dotenv import load_dotenv
load_dotenv()

prompt_data = """
Act as a Shakespeare and write a poem on Generative AI
"""

# Initialize the boto3 client with the specified region
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}
body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    # Ensure that response is retrieved properly
    if 'body' in response:
        response_body = json.loads(response['body'].read())
        
        # Check for the 'generation' key in the response body
        if 'generation' in response_body:
            response_text = response_body['generation']
            print(response_text)
        else:
            print("The key 'generation' was not found in the response body. Here is the response body for debugging:")
            print(response_body)
    else:
        print("No 'body' key found in the response. Here is the response for debugging:")
        print(response)

except boto3.exceptions.Boto3Error as e:
    print(f"An error occurred with boto3: {e}")
except botocore.exceptions.NoRegionError as e:
    print(f"Region not specified: {e}")
except botocore.exceptions.ClientError as e:
    print(f"Client error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
