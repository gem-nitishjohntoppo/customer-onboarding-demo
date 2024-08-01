import streamlit as st
import os
from processing.aws_textract import extract_text
from processing.text_processing import process_large_context_in_chunks,process_large_context
import json
# from text_processing_2 import process_large_context2
# Initialize the Streamlit app
st.title('PortMyHealth_Insurance')

# File uploader allows the user to upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

# Model selection dropdown
models = {
    "Llama 3b 70B": "meta.llama3-70b-instruct-v1:0",
    "Llama 3b 8B": "meta.llama3-8b-instruct-v1:0",
    "Llama 3.1 8B": "meta.llama3-1-8b-instruct-v1:0",
    "Llama 3.1 70B": "meta.llama3-1-70b-instruct-v1:0",
    "Llama 3.1 405B": "meta.llama3-1-405b-instruct-v1:0",
    
}
selected_model = st.selectbox("Select LLM Model", options=list(models.keys()))
print(selected_model)
if uploaded_file is not None:
    # Create the temp directory if it doesn't exist
    temp_dir = "tempDir"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded file to disk
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success('File successfully uploaded.')
    
    # Assuming the schema is static and loaded from a file or defined here
    schema_path = 'schema.json'
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    
    # Extract text from PDF using your AWS Textract function
    context = extract_text(file_path, with_layout=True)
    
    # Process the extracted text
    try:
        model_id = models[selected_model]
        print(f'model selected:{model_id}')
        # if model_id == "gpt-3.5-turbo":
        #     final_json =process_large_context2(context,schema,model_id)

        if model_id =="meta.llama3-1-8b-instruct-v1:0" or model_id =="meta.llama3-1-405b-instruct-v1:0" or model_id =="meta.llama3-1-70b-instruct-v1:0":
            final_json = process_large_context(context,schema,model_id)
        else:    
            final_json = process_large_context_in_chunks(context, schema, model_id)
        print(f'final json :{final_json}')
        st.json(final_json)  # Display JSON output
    except Exception as e:
        st.error(f"Error processing document: {e}")
else:
    st.write("Please upload a PDF file.")
