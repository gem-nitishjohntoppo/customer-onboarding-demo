from aws_textract import extract_text
import json
from text_processing import process_large_context_in_chunks

if __name__ == "__main__":
    file_path = 'schema.json'
    with open(file_path, 'r') as file:
        schema = json.load(file)
    
    context = extract_text('member1_port1 (13).PDF', with_layout=True)
    # print(f'Here:{context}')
    final_json = process_large_context_in_chunks(context, schema,'meta.llama3-70b-instruct-v1:0')
    print(f'ANSWER: {final_json}')
