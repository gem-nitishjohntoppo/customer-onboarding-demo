from groq_api_utils import groq_llm, extract_json_from_string
import json
import time
from bedrock_llm import bedrock_llm
from nltk.tokenize import word_tokenize

# def process_large_context_in_chunks(context, schema):
#     chunks = split_into_chunks(context)
#     # print(f'Chunks here:{chunks}')
#     results = []

#     for chunk in chunks:
#         prompt = generate_prompt(chunk, schema)
#         result = bedrock_llm(prompt, model="meta.llama3-8b-instruct-v1:0")
#         # result1 = str(result)
#         print(f'Bedrock results:{result}')
#         print(f'Bedrock results Type:{type(result)}')

#         parsed_result = json.loads(extract_json_from_string(result))
#         results.append(parsed_result)


#     return merge_results(results)
# def split_into_chunks(context, chunk_size=5000):  # Using a token-based chunk size
#     tokens = word_tokenize(context)
#     print(f'Tokens:{tokens}')
#     return [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
# # def split_into_chunks(context, chunk_size=3000):
# #     return [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
# # def split_into_chunks(context, chunk_size=5000):
# #   val = context.split(' ')
# #   result= [val[i:i+chunk_size] for i in range(0, len(val), chunk_size)]
# #   return [' '.join(string_list) for string_list in result]

# def generate_prompt(chunk, schema):
#     json_schema = json.dumps(schema, indent=4).replace('{', '{{').replace('}', '}}')
#     return f"""Process this chunk: {chunk} with this schema: {json_schema}"""

# def merge_results(results):
#     return {"combined_result": results}

def process_large_context_in_chunks(contexts, schema):
    chunks = [split_into_chunks(entry['context']) for entry in contexts]
    results = []
    for page_chunks in chunks:
        for chunk in page_chunks:
            prompt = generate_prompt(chunk, schema)
            result = bedrock_llm(prompt, model="meta.llama3-70b-instruct-v1:0")
            # result = groq_llm(prompt, model="llama-3.1-405b")
            print(f'Bedrock results: {result}')
            print(f'Bedrock results Type: {type(result)}')
            parsed_result = json.loads(extract_json_from_string(result))
            results.append(parsed_result)
    return merge_results(results,schema)

def split_into_chunks(context, chunk_size=5000):  # Adjusted to handle text as input
    tokens = word_tokenize(context)
    print(f'Tokens:{tokens}')
    return [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

def generate_prompt(chunk, schema):
    json_schema = json.dumps(schema, indent=4).replace('{', '{{').replace('}', '}}')
    print(f'json schema{json_schema}')
    return f"""Process this chunk: {chunk} with this schema. Output Json should contain all categories like this reference {json_schema}.
    If the information provided does not fit into any of the existing categories. Return null json values don't create new new categories."""

def merge_results(results,schema):
    prompt_add = f"""Combine this json : {results} and remove repetitive entites use{schema}."""
    answer =bedrock_llm(prompt_add, model="meta.llama3-70b-instruct-v1:0")
    final_answer = json.loads(extract_json_from_string(answer))
    return {"combined_result": final_answer}