import os
import logging
from typing import Optional, List
import time
from dotenv import load_dotenv
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig

load_dotenv()

SUPPORTED_EXTENSIONS = ("PDF", "JPG", "JPEG", "PNG", "TIF", "TIFF")

logger = logging.getLogger(__name__) 

def _get_extractor_and_bucket():
    
    region = os.getenv("AWS_REGION")
    s3_bucket = os.getenv("S3_BUCKET")

    extractor = Textractor(region_name=region)
    
    return extractor, s3_bucket

def detect(document_path: str, document_bytes: Optional[bytes] = None):
    """Detects text in a local file or from in-memory byte data.
    The document must be in PDF, JPG, PNG, or TIF format.
    """

    if not document_path:
        raise ValueError("Parameter [document_path] must be provided.")

    extension = os.path.splitext(document_path)[1].replace(".", "").upper()  # Fixed the syntax error and added [1] to get the extension
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension [{extension}] - valid extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")

    if not document_bytes:
        with open(document_path, "rb") as f:
            document_bytes = f.read()

    extractor, s3_bucket = _get_extractor_and_bucket()
    

    start_time = time.time()
    if extension in [e for e in SUPPORTED_EXTENSIONS if e != "PDF"]:
        logger.debug(f"Starting synchronous detection for document {document_path} ...")
        document = extractor.detect_document_text(file_source=document_bytes)
        logger.debug(f"Completed synchronous detection for document {document_path} in {time.time() - start_time} seconds")
    else:
        logger.debug(f"Starting asynchronous detection for document {document_path} ...")
        document = extractor.start_document_text_detection(file_source=document_bytes, s3_upload_path=f"s3://{s3_bucket}/textract/", save_image=False)
        logger.debug(f"Completed asynchronous detection for document {document_path} in {time.time() - start_time} seconds")

    return document

def analyze(document_path: str, features: Optional[List[TextractFeatures]] = None, document_bytes: Optional[bytes] = None):
    """Detects text and additional elements, such as forms or tables, in a local file or from in-memory byte data.
    The document must be in PDF, JPG, PNG, or TIF format.
    """
    if not document_path:
        raise ValueError("Parameter [document_path] must be provided.")


    extension = os.path.splitext(document_path)[1].replace(".", "").upper()  # Fixed the syntax error and added [1] to get the extension
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension [{extension}] - valid extensions are: {', '.join(SUPPORTED_EXTENSIONS)}")


    if not document_bytes:
        with open(document_path, "rb") as f:
            document_bytes = f.read()

    features = features or [TextractFeatures.TABLES]
    extractor, s3_bucket = _get_extractor_and_bucket()

    start_time = time.time()

    if extension in [e for e in SUPPORTED_EXTENSIONS if e != "PDF"]:
        logger.debug(f"Starting synchronous analysis for document {document_path} ...")    
        document = extractor.analyze_document(file_source=document_path, features=features)
        logger.debug(f"Completed synchronous detection for document {document_path} in {time.time() - start_time} seconds")
    else:
        logger.debug(f"Starting asynchronous analysis for document {document_path} ...")
        document = extractor.start_document_analysis(file_source=document_bytes, features=features, s3_upload_path=f"s3://{s3_bucket}/textract/", save_image = False)
        logger.debug(f"Completed asynchronous analysis for document {document_path} in {time.time() - start_time} seconds")

    return document



# Function to extract text from a file using AWS Textract
# @st.cache_data
def extract_text(file_path, document_bytes: Optional[bytes] = None, with_layout=True):
    max_retries = 3

    for retry_count in range(max_retries):
        # try:
        if with_layout:
            features_list = [TextractFeatures.TABLES, TextractFeatures.LAYOUT, TextractFeatures.FORMS, TextractFeatures.SIGNATURES]
            document = analyze(file_path, features=features_list, document_bytes=document_bytes)
        else:
            document = detect(file_path, document_bytes=document_bytes)
        
        config = TextLinearizationConfig(
            hide_figure_layout=True,
            title_prefix="# ",
            remove_new_lines_in_leaf_elements= True,
            section_header_prefix="## ",
            table_linearization_format='markdown'#'markdown',
        )

        
        page_results = []
        
        # for i, page in enumerate(document.pages):
        # Limit extraction to the first 5 pages or the total number of pages if less than 5
        num_pages_to_process = min(7, len(document.pages))
        for i in range(num_pages_to_process):
            page = document.pages[i]
            page_text = page.get_text(config=config)
            page_results.append({
                "page": i + 1,
                "context": page_text
            })
        
        return page_results
    
# text_out = extract_text("C:/Users/prashant.solanki/OneDrive - Gemini Solutions/Desktop/Org_Struct/docs2/ex_2_company_4_document_1.pdf", with_layout=True)
# print("TEXT OUTPUT:", text_out)
# for table in table_out:
#     print("TABLE OUTPUT:", table.get_text(TextLinearizationConfig(table_linearization_format='markdown')))

