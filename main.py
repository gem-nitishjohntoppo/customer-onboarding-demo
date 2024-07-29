from aws_textract import extract_text
import json
from text_processing import process_large_context_in_chunks

if __name__ == "__main__":
    file_path = 'schema.json'
    with open(file_path, 'r') as file:
        schema = json.load(file)
    
    context = extract_text('Proposal_Form.pdf', with_layout=True)
    # print(f'Here:{context}')
    final_json = process_large_context_in_chunks(context, schema)
    print(f'ANSWER: {final_json}')
