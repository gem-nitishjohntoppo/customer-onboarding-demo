from groq_api_utils import groq_llm, extract_json_from_string
import json
import time
from bedrock_llm import bedrock_llm
from nltk.tokenize import word_tokenize
from model_selection import model_selection
from llama3_1_8b import llama3_1_8b


def process_large_context_in_chunks(contexts, schema, model_id):
    chunks = [split_into_chunks(entry['context']) for entry in contexts]
    results = []
    for page_chunks in chunks:
        for chunk in page_chunks:
            prompt = generate_prompt(chunk, schema)
            # result = bedrock_llm(prompt, model=model_id)
            result = model_selection(prompt,model= model_id)
            print("JSON being loaded:", result)
            parsed_result = json.loads(extract_json_from_string(result))
            print(f'Parsed results:{parsed_result}')
            results.append(parsed_result)
    return merge_results(results,schema,model_id)

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


def merge_results(results,schema,model_id):
    prompt_add = f"""combine this json : {results} and remove repetitive entites using this{schema}.
    Also check for all null and empty values in json output whether similar results are available in other categories of the schema. 
    Replace any None or empty json value for attribute or category with null json value for that attribute or category.
    Note: Return the output in the corrected json format and only in one json schema"""

    answer = model_selection(prompt_add,model_id)
    # print(f'combine answer{answer}')
    # with open('combine_answer.txt', 'w') as file:
    #     file.write(answer)
    
    final_answer = json.loads(extract_json_from_string(answer))
    # final_answer = extract_json_from_string(answer)
    # with open('extract_json.txt', 'w') as file:
    #     file.write(final_answer)
    
    return {"combined_result": final_answer}

# for llama 3.1
def process_large_context(context, schema, model_id):
    chunk = context
    prompt = generate_prompt(chunk, schema)
    print(f'Prompt is generated')
    result = llama3_1_8b(prompt, model=model_id)
    # result = groq_llm(prompt, model="llama-3.1-405b")
    print(f'Bedrock results: {result}')
    print(f'Bedrock results Type: {type(result)}')
    parsed_result = json.loads(extract_json_from_string(result))
    schema = parsed_result
    # results.append(parsed_result)
    return schema