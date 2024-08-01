from models.llama3_8b import llama3_8b
from models.llama3_70b import llama3_70b


def model_selection(prompt,model):
    with open('prompt.txt', 'w', encoding='utf-8') as file:  # Specify UTF-8 encoding here
        file.write(prompt)
    if model == "meta.llama3-8b-instruct-v1:0":
        return llama3_8b(prompt,model="meta.llama3-8b-instruct-v1:0")
    if model == "meta.llama3-70b-instruct-v1:0":
        return llama3_70b(prompt,model="meta.llama3-70b-instruct-v1:0")
    