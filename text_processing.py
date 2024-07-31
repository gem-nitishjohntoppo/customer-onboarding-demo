from groq_api_utils import groq_llm, extract_json_from_string
import json
import time
from bedrock_llm import bedrock_llm
from nltk.tokenize import word_tokenize


def process_large_context_in_chunks(contexts, schema, model_id):
    chunks = [split_into_chunks(entry['context']) for entry in contexts]
    results = []
    for page_chunks in chunks:
        for chunk in page_chunks:
            prompt = generate_prompt(chunk, schema)
            result = bedrock_llm(prompt, model=model_id)
            # result = groq_llm(prompt, model="llama-3.1-405b")
            print(f'Bedrock results: {result}')
            print("JSON being loaded:", result)
            parsed_result = json.loads(extract_json_from_string(result))
            print(f'Parsed results:{parsed_result}')
            results.append(parsed_result)
    return merge_results(results,schema)

def split_into_chunks(context, chunk_size=5000):  # Adjusted to handle text as input
    tokens = word_tokenize(context)
    print(f'Tokens:{tokens}')
    return [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

def generate_prompt(chunk, schema):
    json_schema = json.dumps(schema, indent=4).replace('{', '{{').replace('}', '}}')
    # print(f'json schema{json_schema}')
    return f"""Process this chunk: {chunk} with this {json_schema}. Output Json should contain all categories like this reference schema.
    Make sure to check null and empty value of output json again on other chunks to find values. 
    If the information provided does not fit into any of the existing categories
    return null json values don't create new categories."""

def merge_results(results,schema):
    prompt_add = f"""Combine this json : {results} and remove repetitive entites using this{schema}.
    Also check for all null and empty values in json output whether similar results are available in other categories of the schema.. 
    Replace any None or empty json value for attribute or category with null json value for that attribute or category.
    Note: Return the output in the corrected json format and only in one json schema"""
    answer =bedrock_llm(prompt_add, model="meta.llama3-70b-instruct-v1:0")
    # print(f'combine answer{answer}')
    # with open('combine_answer.txt', 'w') as file:
    #     file.write(answer)
    
    final_answer = json.loads(extract_json_from_string(answer))
    # final_answer = extract_json_from_string(answer)
    # with open('extract_json.txt', 'w') as file:
    #     file.write(final_answer)
    
    return {"combined_result": final_answer}

# def merge_results(results, schema):
#     # Generate a prompt to process the results using the schema
#     prompt_add = f"""Combine this json : {results} and remove repetitive entites use{schema}."""
#     # Calling the LLM model to process the prompt
#     answer = bedrock_llm(prompt_add, model="meta.llama3-70b-instruct-v1:0")
#     # Extracting a JSON string from the answer
#     json_string = json.loads(extract_json_from_string(answer))
#     # json_string = json_string.replace("'", '"')
#     # json_string = json_string.replace('None', 'null')
#     # json_string = json_string.replace('"null"', 'null')

    
#     # Debugging output
#     print("Extracted JSON string:", json_string)  # Debug statement
    
#     # Trying to load the JSON string to Python object
#     try:
#         final_answer = json.loads((json_string))
#     except json.JSONDecodeError as e:
#         # Handling JSON decoding errors
#         print("Failed to decode JSON:", e)
#         print("Faulty JSON string:", json_string)
#         final_answer = {}  # Defaulting to an empty dictionary if JSON parsing fails

#     # Returning the final result wrapped in a dictionary
#     return {"combined_result": final_answer}
