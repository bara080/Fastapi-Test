"""
Utilities for MongoDB Atlas Vector Search indexing, simple progress tracking,
provider-scoped environment variable loading, and LLM client factory.

This module is intentionally minimal and side-effect free (except where noted):
- No global network calls (the `track_progress` HTTP call is commented out).
- Functions are synchronous and blocking by design.

Do NOT change code structure; documentation and comments only.
"""

from pymongo.errors import OperationFailure
from pymongo.collection import Collection
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import requests  # kept for potential future use in track_progress
from typing import Dict, List
import time
import os
import json
from datetime import datetime
from pathlib import Path
import socket
import os
from dotenv import dotenv_values, find_dotenv

# Polling interval (seconds) for index create/delete/ready loops
SLEEP_TIMER = 5


def create_index(collection: Collection, index_name: str, model: Dict) -> None:
    """
    Create (or recreate) a MongoDB Atlas **Search / Vector Search** index.

    Behavior:
      - Attempts to create the index with the provided `model` definition.
      - If the index already exists, it deletes the existing one and recreates it,
        polling until the deletion is visible via `list_search_indexes()`.

    Parameters
    ----------
    collection : Collection
        Target MongoDB collection (PyMongo Collection object).
    index_name : str
        Name of the search index to create.
    model : Dict
        Index definition document as expected by Atlas Search
        (e.g., for vectorSearch: fields[].type/path/numDimensions/similarity).

    Raises
    ------
    Exception
        Re-raises unexpected exceptions during drop/recreate path with context.

    Notes
    -----
    - Uses `collection.create_search_index(model=...)` (PyMongo 4.11+).
    - Uses `collection.drop_search_index(name=...)` followed by polling
      `collection.list_search_indexes()` until the target name disappears.
    - Blocking call; polling cadence controlled by `SLEEP_TIMER`.
    """
    try:
        print(f"Creating the {index_name} index")
        collection.create_search_index(model=model)
    except OperationFailure:
        print(f"{index_name} index already exists, recreating...")
        try:
            print(f"Dropping {index_name} index")
            collection.drop_search_index(name=index_name)

            # Poll for index deletion to complete (eventual consistency safety)
            while True:
                indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get("name") == index_name for idx in indexes)
                if not index_exists:
                    print(f"{index_name} index deletion complete")
                    break
                print(f"Waiting for {index_name} index deletion to complete...")
                time.sleep(SLEEP_TIMER)

            print(f"Creating new {index_name} index")
            collection.create_search_index(model=model)
            print(f"Successfully recreated the {index_name} index")
        except Exception as e:
            raise Exception(f"Error during index recreation: {str(e)}")


def check_index_ready(collection: Collection, index_name: str) -> None:
    """
    Block until the specified Search/Vector Search index reports status **READY**.

    Parameters
    ----------
    collection : Collection
        Target MongoDB collection (PyMongo Collection object).
    index_name : str
        Name of the search index to check.

    Notes
    -----
    - Loops on `collection.list_search_indexes()` and inspects:
        * presence of `index_name`
        * `index['status'] == 'READY'`
    - Prints intermediate statuses (e.g., "BUILDING").
    - Poll interval controlled by `SLEEP_TIMER`.
    """
    while True:
        indexes = list(collection.list_search_indexes())
        matching_indexes = [idx for idx in indexes if idx.get("name") == index_name]

        if not matching_indexes:
            print(f"{index_name} index not found")
            time.sleep(SLEEP_TIMER)
            continue

        index = matching_indexes[0]
        status = index["status"]
        if status == "READY":
            print(f"{index_name} index status: READY")
            print(f"{index_name} index definition: {index['latestDefinition']}")
            break

        print(f"{index_name} index status: {status}")
        time.sleep(SLEEP_TIMER)

def track_progress(task: str, workshop_id: str) -> None:
    """
    Track progress locally (no network).
    Appends a JSON line to a local log file.

    Env (optional):
      PROGRESS_LOG_PATH = path to log file (default: ./logs/progress.jsonl)
    """
    import json
    from datetime import datetime
    from pathlib import Path
    import socket
    import os

    log_path = os.environ.get("PROGRESS_LOG_PATH", "./logs/progress.jsonl")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": datetime.utcnow().isoformat(),
        "task": task,
        "workshop_id": workshop_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[progress] {task} ({workshop_id}) → {log_path}")
    except Exception as e:
        print(f"[progress][error] failed to write log: {e}")


def set_env(providers: List[str], passkey: str = "", env_path: str | None = None) -> Dict[str, str]:
    """
    Export environment variables scoped by provider prefixes from a `.env` file.

    Selection rule:
      Only keys that start with `<PROVIDER>_` (uppercased) are exported into
      `os.environ`. This avoids leaking unrelated secrets.

    Parameters
    ----------
    providers : List[str]
        Provider slugs, e.g. ["openai", "azure_openai", "google", "voyageai"].
        Each slug maps to an uppercased prefix `<SLUG>_`.
    passkey : str, optional
        Unused here (reserved for workshop flows that gate access). Kept for API
        compatibility; safe to ignore.
    env_path : str | None, optional
        Explicit path to a `.env` file. If None, uses `find_dotenv(usecwd=True)`.

    Returns
    -------
    Dict[str, str]
        Mapping of keys applied into the process environment.

    Expected .env Keys
    ------------------
    - OPENAI_API_KEY=...
    - AZURE_OPENAI_ENDPOINT=...
    - AZURE_OPENAI_API_KEY=...
    - GOOGLE_API_KEY=...
    - VOYAGEAI_API_KEY=...
    - (any other `<PROVIDER>_*` you require)

    Example
    -------
    >>> set_env(["openai", "azure_openai", "google", "voyageai"])
    {'OPENAI_API_KEY': '***', 'AZURE_OPENAI_ENDPOINT': '***', ...}
    """
    path = env_path or find_dotenv(usecwd=True)
    env = dotenv_values(path)
    applied: Dict[str, str] = {}

    for provider in providers:
        prefix = f"{provider.upper()}_"
        for k, v in env.items():
            if k.startswith(prefix) and v is not None:
                os.environ[k] = v
                applied[k] = v

    return applied


def get_llm(provider: str):
    """
    Factory for chat LLM clients across multiple providers.

    Parameters
    ----------
    provider : str
        One of: 'openai', 'aws', 'google', 'microsoft'.

    Returns
    -------
    Any
        A LangChain-compatible chat model instance.

    Models
    ------
    - openai    → ChatOpenAI(model="gpt-4o", temperature=0)
    - aws       → ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", region "us-west-2")
    - google    → ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    - microsoft → AzureChatOpenAI(azure_endpoint=..., azure_deployment="gpt-4o", api_version="2023-06-01-preview")

    Raises
    ------
    Exception
        If the provider is not one of the supported values.

    Notes
    -----
    - Ensure the corresponding environment variables/credentials are set:
        * OPENAI_API_KEY
        * AWS credentials (if using Bedrock) via standard AWS env/roles
        * GOOGLE_API_KEY
        * AZURE_* (endpoint, key, deployment, api version)
    """
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0)
    elif provider == "aws":
        return ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            model_kwargs=dict(temperature=0),
            region_name="us-west-2",
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
        )
    elif provider == "microsoft":
        return AzureChatOpenAI(
            azure_endpoint="https://gai-326.openai.azure.com/",
            azure_deployment="gpt-4o",
            api_version="2023-06-01-preview",
            temperature=0,
        )
    else:
        raise Exception("Unsupported provider. provider can be one of 'openai', 'aws', 'google', 'microsoft'.")
