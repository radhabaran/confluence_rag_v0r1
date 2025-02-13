# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
api_key = os.environ['OA_API']
os.environ['OPENAI_API_KEY'] = api_key

# Huggingface API configuration
hf_api_key = os.environ['MY_HF_TOKEN1']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_api_key

class Config:
    # API Key for OpenAI
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    # API Key for Huggingface
    HF_API_KEY = os.environ['HUGGINGFACEHUB_API_TOKEN']

    # API Key for Huggingface

    # Confluence settings
    CONFLUENCE_USERNAME = 'radhabaran.mohanty@gmail.com'
    CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_KEY')  # Load from environment variable
    CONFLUENCE_PAGE_ID = '98319'  # The ID of the page you want to access

    # Collection settings
    COLLECTION_NAME = "knowledge_base"

    # File paths
    DOCUMENT_DIRECTORY = "./data/documents/"  # Main directory for all documents
    LOCAL_QDRANT_PATH = "./data/local_qdrant"

    # Document type directories (subdirectories)
    PDF_DIRECTORY = os.path.join(DOCUMENT_DIRECTORY, "pdfs")
    PPT_DIRECTORY = os.path.join(DOCUMENT_DIRECTORY, "presentations")

    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.pdf', '.ppt', '.pptx']

    # Search settings
    SEARCH_LIMIT = 10
    SIMILARITY_THRESHOLD = 0.7

    # Chunking configuration
    CHUNK_SIZE = 512  # Adjust as needed
    CHUNK_OVERLAP = 50  # Adjust as needed
    BATCH_SIZE = 100  # Adjust as needed