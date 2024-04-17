import json
import os
import pathlib

import requests
from langchain_core.documents import Document

from aiasan import utils, vectorstore

log = utils.logger(__name__)


def readfile(path: str, dir_path: pathlib.Path) -> str:
    """Read the content of a markdown file. Use listdir tool to find available path"""
    return (dir_path / path).read_text()


def listdir(dir_path: pathlib.Path) -> list[str]:
    """List all markdown files in the current directory and recursively in sub-directories"""
    return [str(d.relative_to(dir_path)) for d in dir_path.rglob("*.md")]


def search(keyword: str) -> str:
    """Leverages Google Search to retrieve web search results for the provided keyword."""
    log.info(f"Google search: {keyword}")
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": keyword})

    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text


def retrieve_documents(
    subject: str, db: vectorstore.VectorStore, k: int = 3
) -> list[Document]:
    """Utilizes a powerful document retrieval system to find the k most relevant piece of notes within your existing notes that address the specified subject."""
    retriever = db.vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(subject)
    return docs
