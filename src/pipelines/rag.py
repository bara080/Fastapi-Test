# import os
# import sys
# from utils import track_progress, set_env
# import json
# from pymongo import MongoClient
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from typing import Dict, List
# import voyageai
# from tqdm import tqdm
# import requests
# from utils import create_index, check_index_ready
# from datetime import datetime

# # TODO: Setup prerequisites

# sys.path.append(os.path.join(os.path.dirname(os.getcwd())))


# # connect to DB
# MONGODB_URI = os.environ.get("MONGODB_URI")

# Mongodb_client = MongoClient(MONGODB_URI)

# Mongodb_client.admin.command("ping")

# print(f" DB is connected and also pingged successfully \n")


# # Tracj progress of key steps


# track_progress("cluster_creation", "ai_rag_lab")

# # Set passkey for AI
# PASSKEY = "PASS-KEY-VOAYGE-AI"

# # OBTAIN A VOYAGE API KEY

# set_env(["VOYAGE_API_KEY"], PASSKEY)


# # Load dataset

# DATA_PATH = "../data/mongodb_doc.json"
# with open(DATA_PATH, "r") as data_file:
#     json_data = data_file.read()
# doc = json.loads(json_data)


# # Note the length of documents in the datasets

# len(doc)

# # preview a document to understand its structure

# doc[0]

# # TODO  :   CHUNK AND EMBED THE DATA

# # what is data is already embedded
# # Common list of separators for text data

# separators = ["\n\n", "\n", " ", "", "#", "##", "###"]





# # use the `RecursiveCharacterTextzSplitter` from langchain to first split a piece of text of 'sepearator ` above
# # Then recursively merge them into tokens until the specified chunk is reached
# # For text data, you typically want to keep 1-2 paragraphs (~200 tokens) in a single chunk
# # Set chunk overlap to 0 for contextualized embedding, otherwise 15-20 of chunk  size
# # The `model_name` parameter indicates which encoder to use for tokenization, in this case GPT-4's encoder


# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     model_name="gpt-4", separators=separators, chunk_size= 200, chunk_overlap=0
# )  


# def get_chunk (doc: Dict, text_field: str) -> List[Dict]:
#     """
#     Chunk up document

#     Args:
#         doc (Dict): Parent document to generate chunks from
#         text_filed (str): Text field to chunk

#     Returns:
#         List[Dict]: List of chunked documents
#     """
#     # Extract the field to chunk from `doc`
#     text = doc [text_field]
#     # Split `text` using `split_text` method of the `text_splitter` object above
#     chunks = text_splitter(text)
    
# # Initialize the Voyage Ai client
# vo = voyageai.client()

# def get_embeddings(content: List[str], input_type: str) -> List[float] | List[List[float]]:
#     """
#     Get contextualized embeddings for each chunk.

#     Args:
#         content (List[str]): List of chunked texts or the user query as a list
#         input_type (str): Type of input, either "document" or "query" 

#     Returns:
#         List[float] | List[List[float]]: Contextualized embeddings
#     """
#     # Use the `contextualized_embed` method of the Voyage AI API to get contextualized embeddings for each chunk with the following arguments:
#     # inputs: `content` wrapped in another list
#     # model: `voyage-context-3`
#     # input_type: `input_type`
#     embds_obj = vo.contextualized_embed(inputs=[content], model="voyage-context-3", input_type=input_type)
#      # If `input_type` is "document", there is a single result with multiple embeddings, one for each chunk
#     if input_type == "document":
#         embeddings = [emb for r in embds_obj.results for emb in r.embeddings]
#     # If `input_type` is "query", there is a single result with a single embedding
#     if input_type == "query":
#         embeddings = embds_obj.results[0].embeddings[0]
#     return embeddings
    
# embedded_docs = []
# # Iterate through `docs` from Step 2
# for doc in tqdm(docs):
#     # Use the `get_chunks` function to chunk up the "body" field in each document
#     chunks = get_chunks(doc, "body")
#     # Pass all the `chunks` to the `get_embeddings` function to get contextualized embeddings for each chunk
#     # `input_type` should be set to "document" since we are embedding the "documents" for RAG
#     chunk_embeddings = get_embeddings(chunks, "document")
#     # For each chunk, create a new document with the original metadata
#     # Replace the `body` with the chunk content and add an `embedding` field
#     for chunk, embedding in zip(chunks, chunk_embeddings):
#         # Create a new document by copying the original document
#         chunk_doc = doc.copy()
#         # Replace the `body` field of `chunk_doc` with the chunk content
#         chunk_doc["body"] = chunk
#         # Add an `embedding` field to `chunk_doc`, containing the embedding for this chunk
#         chunk_doc["embedding"] = embedding
#         # Append `chunk_doc` to `embedded_docs`
#         embedded_docs.append(chunk_doc)



# # Notice that the length of `embedded_docs` is greater than the length of `docs` from Step 2 above
# # This is because each document in `docs` has been split into multiple chunks
# len(embedded_docs)


# # Preview a chunked document to understand its structure
# # Note that the structure looks similar to the original docs, except the `body` field now contains smaller chunks of text
# # Each document also has an additional `embedding` field
# embedded_docs[0]


import os
import sys
from utils import track_progress, set_env
import json
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, List
import voyageai
from tqdm import tqdm
import requests
from utils import create_index, check_index_ready
from datetime import datetime
import dotenv 
# ============================================
# TODO :  SETUP ENVIRONMENT VARIABLES
# Add parent DIR to path to import utils
# ============================================

sys.path.append(os.path.join(os.path.dirname(os.getcwd())))

dotenv.load_dotenv()

print(f"dotenv loaded correclty")


# --------------------------------------------
# Connect to MongoDB
# --------------------------------------------

MONGO_URL = os.getenv("MONGODB_URL")
print(f"MONGO URL LOADED")
#exit(1)

# Initialize MongoDB python client
mongodb_client = MongoClient(MONGO_URL)
print(f"Connection for mongo client established")
#exit(1)
# Check the connection to the server
#mongodb_client.admin.command("ping")




# Assuming 'client' is your established MongoClient instance
# For example:
#client = MongoClient("mongodb://localhost:27017/") 

try:
    # Send a ping to confirm a successful connection
    mongodb_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error during ping: {e}")
# finally:
#     # Close the connection when done (optional, depending on your application lifecycle)
#     mongodb_client


print(f"MOngog is being pinged successfully")
#exit(1)

print("\n\n")


# Tracj progress of key steps

track_progress("cluster_creation", "ai_rag_lab")

# Set passkey for AI
PASSKEY = "PASS-KEY-VOAYGE-AI"

# OBTAIN A VOYAGE API KEY

set_env(["VOYAGE_API_KEY"], PASSKEY)


# Load dataset
# Load the data from the JSON file
with open("./data/mongodb_docs.json", "r") as data_file:
    book_data = data_file.read()

doc = json.loads(book_data)

# DATA_PATH = "./data/mongodb_doc.json"
# with open(DATA_PATH, "r") as data_file:
#     json_data = data_file.read()
# doc = json.loads(json_data)


# Note the length of documents in the datasets

len(doc)

# preview a document to understand its structure

doc[0]

# TODO  :   CHUNK AND EMBED THE DATA

# what is data is already embedded
# Common list of separators for text data

separators = ["\n\n", "\n", " ", "", "#", "##", "###"]


# use the `RecursiveCharacterTextzSplitter` from langchain to first split a piece of text of 'sepearator ` above
# Then recursively merge them into tokens until the specified chunk is reached
# For text data, you typically want to keep 1-2 paragraphs (~200 tokens) in a single chunk
# Set chunk overlap to 0 for contextualized embedding, otherwise 15-20 of chunk  size
# The `model_name` parameter indicates which encoder to use for tokenization, in this case GPT-4's encoder

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4", separators=separators, chunk_size=200, chunk_overlap=0
)  


def get_chunk(doc: Dict, text_field: str) -> List[str]:
    """
    Chunk up document

    Args:
        doc (Dict): Parent document to generate chunks from
        text_filed (str): Text field to chunk

    Returns:
        List[Dict]: List of chunked documents
    """
    # Extract the field to chunk from `doc`
    text = doc[text_field]
    # Split `text` using `split_text` method of the `text_splitter` object above
    chunks = text_splitter.split_text(text)
    return chunks
    

# Initialize the Voyage Ai client
vo = voyageai.Client()

def get_embeddings(content: List[str], input_type: str) -> List[float] | List[List[float]]:
    """
    Get contextualized embeddings for each chunk.

    Args:
        content (List[str]): List of chunked texts or the user query as a list
        input_type (str): Type of input, either "document" or "query" 

    Returns:
        List[float] | List[List[float]]: Contextualized embeddings
    """
    # Use the `contextualized_embed` method of the Voyage AI API to get contextualized embeddings for each chunk with the following arguments:
    # inputs: `content` wrapped in another list
    # model: `voyage-context-3`
    # input_type: `input_type`
    embds_obj = vo.contextualized_embed(inputs=content, model="voyage-context-3", input_type=input_type)
    # If `input_type` is "document", there is a single result with multiple embeddings, one for each chunk
    if input_type == "document":
        embeddings = [emb for r in embds_obj.results for emb in r.embeddings]
    # If `input_type` is "query", there is a single result with a single embedding
    if input_type == "query":
        embeddings = embds_obj.results[0].embeddings[0]
    return embeddings
    

embedded_docs = []
# Iterate through `docs` from Step 2
docs = doc
for doc in tqdm(docs):
    # Use the `get_chunks` function to chunk up the "body" field in each document
    chunks = get_chunk(doc, "body")
    # Pass all the `chunks` to the `get_embeddings` function to get contextualized embeddings for each chunk
    # `input_type` should be set to "document" since we are embedding the "documents" for RAG
    chunk_embeddings = get_embeddings(chunks, "document")
    # For each chunk, create a new document with the original metadata
    # Replace the `body` with the chunk content and add an `embedding` field
    for chunk, embedding in zip(chunks, chunk_embeddings):
        # Create a new document by copying the original document
        chunk_doc = doc.copy()
        # Replace the `body` field of `chunk_doc` with the chunk content
        chunk_doc["body"] = chunk
        # Add an `embedding` field to `chunk_doc`, containing the embedding for this chunk
        chunk_doc["embedding"] = embedding
        # Append `chunk_doc` to `embedded_docs`
        embedded_docs.append(chunk_doc)



# Notice that the length of `embedded_docs` is greater than the length of `docs` from Step 2 above
# This is because each document in `docs` has been split into multiple chunks
len(embedded_docs)


# Preview a chunked document to understand its structure
# Note that the structure looks similar to the original docs, except the `body` field now contains smaller chunks of text
# Each document also has an additional `embedding` field
embedded_docs[0]
