from groq_api_utils import groq_llm, extract_json_from_string
import json
import time

def process_large_context_in_chunks(context, schema):
    chunks = split_into_chunks(context)
    results = []


    for chunk in chunks:
        prompt = generate_prompt(chunk, schema)
        result = groq_llm(prompt, model="llama3-8b-8192")
        parsed_result = json.loads(extract_json_from_string(result))
        results.append(parsed_result)


    return merge_results(results)

def split_into_chunks(context, chunk_size=5000):
    return [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

def generate_prompt(chunk, schema):
    json_schema = json.dumps(schema, indent=4).replace('{', '{{').replace('}', '}}')
    return f"""Process this chunk: {chunk} with this schema: {json_schema}"""

def merge_results(results):
    return {"combined_result": results}