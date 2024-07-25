import streamlit as st
import os
from aws_textract import extract_text
from text_processing import process_large_context_in_chunks
import json

# Initialize the Streamlit app
st.title('PDF to JSON Processor')

# File uploader allows the user to upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

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
        final_json = process_large_context_in_chunks(context, schema)
        st.json(final_json)  # Display JSON output in a pretty format
    except Exception as e:
        st.error(f"Error processing document: {e}")
else:
    st.write("Please upload a PDF file.")
