# Imports
import os
import sys
import json
from typing import List, Dict, Optional

from pymongo import MongoClient
from PIL import Image
from tqdm import tqdm
import dotenv
import requests
import time
import voyageai

from utils import set_env, create_index,check_index_ready, utils

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

# Track progress of the key steps
#utils("cluster-creation", "ai_vs_lab")

# --------------------------------------------
# TODO: Set the passkey provided by your workshop instructor
# --------------------------------------------

PASSKEY = "VOYAGE_API_KEY"  # <- replace with the actual passkey token string

# Obtain a Voyage AI key from AI model proxy

set_env(["VOYAGE_API_KEY"], PASSKEY)

# ============================================
# TODO : IMPORT DATA INTO MONGODB
# ============================================

# Database name
DB_NAME = "zinga-General-purpose"
# Collection name
COLLECTION_NAME = "books"
# Name of the vector search index
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Connect to the collection
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

print(f"Connected to collection: {COLLECTION_NAME} in database: {DB_NAME}")
#

# Load the data from the JSON file
with open("./data/books.json", "r") as data_file:
    book_data = data_file.read()

data = json.loads(book_data)
print(f"Total records to be inserted into the collection {COLLECTION_NAME}: {len(data)}")

collection.delete_many({})
collection.insert_many(data)

print(f"{collection.count_documents({})} documents ingested into the {COLLECTION_NAME} collection.")
print("\n\n")


# ============================================
# TODO:  GENERATING EMBEDDINGS
# ============================================

# Initialize Voyage AI client
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Example: Image embedding
image_url = "https://images.isbndb.com/covers/4318463482198.jpg"

# Load the image from the URL
image = Image.open(requests.get(image_url, stream=True).raw)

# Use the 'multimodal_embed' method of the Voyage AI API to embed the image
# Inputs: the image wrapped in a list of lists; model: 'voyage-multimodal-3'
resp = vo.multimodal_embed(inputs=[[image]], model="voyage-multimodal-3", input_type="document")

# Print the embedding
print("\n\n")
print(f"Embedding length (image): {len(resp.embeddings[0])}")
print("\n\n")

# Extract the first embedding from the list
embedding = resp.embeddings[0]

# Example: Text embedding
text = "Puppy Preschool: Raising your puppy right—from the start!"
resp_text = vo.multimodal_embed(inputs=[[text]], model="voyage-multimodal-3", input_type="query")
print(f"Embedding length (text): {len(resp_text.embeddings[0])}")
print("\n\n")

# Get the embeddings as a list from the response
_ = resp_text.embeddings[0]

def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for a given text using Voyage AI.

    Args:
        text (str): Input text.

    Returns:
        List[float]: Embedding vector for the input text.
    """
    out = vo.multimodal_embed(inputs=[[text]], model="voyage-multimodal-3", input_type="document")
    return out.embeddings[0]

# ======================================================
# TODO: ADDING EMBEDDINGS TO EXISTING DATA IN ATLAS
# ======================================================

# Field in the documents to embed—in this case, the book cover (image URL)
field_to_embed = "cover"

# Name of the embedding field to add to the documents
embedding_field = "embedding"

def get_embeddings(content: str, mode: str, input_type: str) -> List[float]:
    """
    Get embeddings for the given content using Voyage AI.

    Args:
        content (str): The content to embed (text or image path/URL).
        mode (str): 'text' or 'image'.
        input_type (str): 'document' or 'query'.

    Returns:
        List[float]: The embedding vector.
    """
    if mode == "image":
        if content.startswith("http"):
            content = Image.open(requests.get(content, stream=True).raw)
        else:
            content = Image.open(content)
        out = vo.multimodal_embed(inputs=[[content]], model="voyage-multimodal-3", input_type=input_type)
    else:
        # text mode
        out = vo.multimodal_embed(inputs=[[content]], model="voyage-multimodal-3", input_type=input_type)

    return out.embeddings[0]

# Update each document in the `collection` with embeddings
print("\n\n")
results = collection.find({})
for result in tqdm(results, desc="Embedding documents"):
    content = result.get(field_to_embed)
    if not content:
        continue
    # Embed the cover image as a document vector
    emb = get_embeddings(content, "image", "document")
    # Update the document with the computed embedding
    collection.update_one({"_id": result["_id"]}, {"$set": {embedding_field: emb}})

def add_embeddings_to_mongodb(collection, batch_size: int = 100):
    """
    Add embeddings to existing documents in MongoDB in batches.

    Args:
        collection: PyMongo collection handle.
        batch_size (int, optional): Batch size for processing. Defaults to 100.
    """
    cursor = collection.find({embedding_field: {"$exists": False}}, {"_id": 1, field_to_embed: 1})
    batch = []
    for doc in cursor:
        content = doc.get(field_to_embed)
        if not content:
            continue
        emb = get_embeddings(content, "image", "document")
        batch.append(
            {
                "filter": {"_id": doc["_id"]},
                "update": {"$set": {embedding_field: emb}},
            }
        )
        if len(batch) >= batch_size:
            for op in batch:
                collection.update_one(op["filter"], op["update"])
            batch = []
    if batch:
        for op in batch:
            collection.update_one(op["filter"], op["update"])

# ============================================
# TODO: CREATE A VECTOR SEARCH INDEX
# ============================================

"""
Create vector index definition specifying:
- path: path to the embedding field
- numDimensions: number of embedding dimensions (depends on the embedding model)
- similarity: similarity metric ('cosine', 'euclidean', 'dotProduct', 'manhattan')
"""

model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": embedding_field,
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

# Create the index
create_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# Verify index is READY
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# Track progress of key steps-- DO NOT CHANGE
#utils("vs_index_creation", "ai_vs_lab")

# ============================================
# TODO : PERFORMING VECTOR SEARCH QUERIES
# ============================================

def vector_search(user_query: str, mode: str, filter: Optional[Dict] = None) -> None:
    """
    Perform a vector search for the given query in the specified collection.

    Args:
        user_query (str): The query string (text or image URL/path).
        mode (str): 'text' or 'image' for query embedding mode.
        filter (Optional[Dict]): Optional Atlas filter to pre-filter documents.

    Returns:
        None
    """
    # Generate query embedding
    query_embedding = get_embeddings(user_query, mode, "query")

    pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                "queryVector": query_embedding,
                "path": embedding_field,
                "numCandidates": 20,
                "filter": filter,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "cover": 1,
                "year": 1,
                "pages": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = collection.aggregate(pipeline)

    # Print book title, score, and cover image
    for book in results:
        cover_url = book.get("cover")
        try:
            cover = Image.open(requests.get(cover_url, stream=True).raw).resize((100, 150))
        except Exception:
            cover = None
        print(f"{book.get('title')} ({book.get('year')}, {book.get('pages')} pages): {book.get('score')}")
        if cover:
            try:
                from IPython.display import display  # optional for notebooks
                display(cover)
            except Exception:
                pass

# Test the vector search with a text query
vector_search("A man wearing a golden crown", "text")

# Also try these text queries:
# - A rainbow of lively colors
# - Creatures wondrous or familiar
# - A boy and the ocean
# - Houses

# Test the vector search with an image query
vector_search("https://images.isbndb.com/covers/10835953482746.jpg", "image")

# Also try these image queries:
# - ../data/images/salad.jpg
# - ../data/images/kitten.png
# - ../data/images/barn.png

# ============================================
# TODO : ADDING PRE-FILTERS TO VECTOR SEARCH
# ============================================

# Modify the vector search index model to include the `year` field as a filter
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": embedding_field,
                "numDimensions": 1024,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "year"},
        ]
    },
}

# Re-create the vector search index with the modified model
create_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# Create a filter definition: books where year >= 2002
flt = {"year": {"$gte": 2002}}
vector_search("A boy and the ocean", "text", flt)

# ============================================
# TODO: ENABLE VECTOR QUANTIZATION
# ============================================

# Modify the vector search index model to use scalar quantization
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": embedding_field,
                "numDimensions": 1024,
                "similarity": "cosine",
                "quantization": "scalar",
            },
        ]
    },
}

# Re-create the vector search index with quantization
create_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# ============================================
# HYBRID SEARCH
# ============================================

# Name of the full-text search index
ATLAS_FTS_INDEX_NAME = "fts_index"

# Create full-text search index definition specifying the field mappings
model = {
    "name": ATLAS_FTS_INDEX_NAME,
    "type": "search",
    "definition": {
        "mappings": {"dynamic": False, "fields": {"title": {"type": "string"}}}
    },
}

# Create FTS index
print(f"Creating index for with specifying fields")
create_index(collection, ATLAS_FTS_INDEX_NAME, model)
print(f"Creating index for with specifying fields, {create_index}")
print("\n\n")

# Reset the vector search index to the original definition (no filters/quantization)
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": embedding_field,
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

print("\n\n")
# Reset vector index
create_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)
print(f"Creating index for with NO fields, {create_index}")
print("\n\n")

check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)
print(f"Creating index for with NO fields, {check_index_ready}")
print("\n\n")

check_index_ready(collection, ATLAS_FTS_INDEX_NAME)
print(f"Creating index for with NO fields, {check_index_ready}")
print("\n\n")


def hybrid_search(user_query: str, vector_weight: float, full_text_weight: float) -> None:
    """
    Retrieve relevant documents for a user query using hybrid search.

    Args:
        user_query (str): User query string (text).
        vector_weight (float): Weight of vector search in final score.
        full_text_weight (float): Weight of full-text search in final score.
    """
    pipeline = [
        # Vector search stage
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                "path": embedding_field,
                "queryVector": get_embeddings(user_query, "text", "query"),
                "numCandidates": 20,
                "limit": 10,
            }
        },
        # Group all documents returned by the vector search into a single array named `docs`
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        # Unwind docs; store position as rank
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        # Reciprocal-rank score for vector search
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 60]}]},
                    ]
                }
            }
        },
        # Keep only required fields
        {
            "$project": {
                "vs_score": 1,
                "_id": "$docs._id",
                "title": "$docs.title",
                "cover": "$docs.cover",
            }
        },
        # Union with a full-text search stage
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    # Full-text search
                    {
                        "$search": {
                            "index": ATLAS_FTS_INDEX_NAME,
                            "text": {"query": user_query, "path": "title"},
                        }
                    },
                    {"$limit": 10},
                    {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {"$divide": [1.0, {"$add": ["$rank", 60]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "title": "$docs.title",
                            "cover": "$docs.cover",
                        }
                    },
                ],
            }
        },
        # Combine scores
        {
            "$group": {
                "_id": "$_id",
                "title": {"$first": "$title"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
                "cover": {"$first": "$cover"},
            }
        },
        {
            "$project": {
                "_id": 1,
                "title": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
                "cover": 1,
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "title": 1,
                "vs_score": 1,
                "fts_score": 1,
                "cover": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": 5},
    ]

    results = collection.aggregate(pipeline)

    # Print book title, scores, and cover image
    for book in results:
        print(
            f"{book.get('title')}, VS Score: {book.get('vs_score')}, FTS Score: {book.get('fts_score')}"
        )
        try:
            cover = Image.open(requests.get(book["cover"], stream=True).raw).resize((100, 150))
            try:
                from IPython.display import display
                display(cover)
            except Exception:
                pass
        except Exception:
            pass

# Test the hybrid search queries
hybrid_search(
    user_query="My Favorite Summer",
    vector_weight=1.0,
    full_text_weight=0.0,
)

hybrid_search(
    user_query="My Favorite Summer",
    vector_weight=0.3,
    full_text_weight=0.7,
)

# Close connection
