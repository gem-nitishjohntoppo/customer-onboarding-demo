from groq import Groq, RateLimitError
import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

def groq_llm(input_prompt, model='llama-3.1-405b'):
    api_key = os.getenv("GROQ_API")
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": "Your Task is to correct the given json in the json format provided and return the corrected Json."
        },        
        {
            "role": "user",
            "content": input_prompt,
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    return chat_completion.choices[0].message.content


def extract_json_from_string(data_str):
    pattern = r'\{.*\}'
    match = re.search(pattern, data_str, re.DOTALL)
    json_data = match.group() if match else None
    return json_data
