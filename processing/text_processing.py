from models.groq_api_utils import groq_llm, extract_json_from_string
import json
from models.bedrock_llm import bedrock_llm
from nltk.tokenize import word_tokenize
from processing.model_selection import model_selection
from models.llama3_1_8b import llama3_1_8b
import json_repair

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
    return f'''Generate a valid JSON output based on the following: 
    Input:
    
        
    A JSON schema definition: `{json_schema}`    
    
    Text chunks containing relevant information: `{chunk}` 
    Output:
    
    A single, well-formed JSON object adhering to the provided schema.    
    
    No additional notes or explanations. 
    Rules:
    
    Structure: The output JSON must strictly match the schema's structure. No keys from the schema should be omitted.   
    
    Arrays: If the schema defines an array of JSON objects, each object within the array must have an identical structure.
    
    Missing/Empty Values: For fields not found in the text chunks or present as empty strings (""), assign the value `null`.   
    
    Contextual Filling: Use the information from the text chunks to populate the key-value pairs in the JSON output.'''


def merge_results(results,schema,model_id):
    json_schema = json.dumps(schema, indent=4).replace('{', '{{').replace('}', '}}')
    # prompt_add = f"""combine this json : {results} and remove repetitive entites using this{schema}.
    # Also check for all null and empty values in json output whether similar results are available in other categories of the schema. 
    # Replace any None or empty json value for attribute or category with null json value for that attribute or category.
    # Note: Return the output in the corrected json format and only in one json schema"""
    prompt_add =f"""Combine the following JSON objects: {results} into a single cohesive JSON object, ensuring that there are no repetitive attributes. Use the provided schema {json_schema} as a reference to guide the structure of the final JSON object.

Instructions:
1. Consolidate all JSON objects from the list into one unified JSON object.
2. Remove any repetitive attributes, ensuring each attribute appears only once in the final JSON object.
3. For attributes that have null or empty values, check across all JSON objects for similar attributes with non-null values and use those values if available.
4. Ensure that any attribute or category with consistently null or empty values across all JSON objects is assigned a null value in the final JSON object.
5. The final output should strictly adhere to the provided schema, with no additional or missing attributes.
6. Return the consolidated JSON object in a well-formed and valid JSON format.

Output the final corrected JSON object."""


    answer = model_selection(prompt_add,model_id)
    print(f'combine answer{answer}')
    # with open('combine_answer.txt', 'w') as file:
    #     file.write(answer)
    
    # final_answer = json.loads(extract_json_from_string(answer))
    final_answer = extract_json_from_string(answer)
    final_answer=json_repair.loads(final_answer)
    # with open('extract_json.txt', 'w') as file:
    #    json.dump(final_answer,file)
    
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
    parsed_result = extract_json_from_string(result)
    parsed_result=json_repair.loads(parsed_result)
    schema = parsed_result
    # results.append(parsed_result)
    return schema