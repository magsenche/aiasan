import pathlib
import pickle
import uuid
from typing import Optional

from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings


def get_ids(l: list):
    return [str(uuid.uuid4()) for _ in l]


class VectorStore:
    splitter = MarkdownTextSplitter()

    def __init__(
        self,
        store_path: pathlib.Path,
        embedding: Embeddings,
        vectorstore: Optional[FAISS] = None,
        source_to_docstore_id: dict[str] = {},
    ):
        self.store_path = store_path
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.source_to_docstore_id = source_to_docstore_id

    @classmethod
    def initialize(
        cls, store_path: pathlib.Path, embedding: Embeddings, docs_path: list[str]
    ):
        loader = DirectoryLoader(
            docs_path, "**/*.md", loader_cls=UnstructuredMarkdownLoader
        )
        docs = loader.load()
        splitted_docs = cls.splitter.split_documents(docs)
        doc_ids = get_ids(splitted_docs)
        vectorstore = FAISS.from_documents(
            documents=splitted_docs, embedding=embedding, ids=doc_ids
        )
        source_to_docstore_id = {}
        for doc, id in zip(splitted_docs, doc_ids):
            source = str(pathlib.Path(doc.metadata["source"]).relative_to(docs_path))
            if source in source_to_docstore_id:
                source_to_docstore_id[source].append(id)
            else:
                source_to_docstore_id[source] = [id]
        return cls(store_path, embedding, vectorstore, source_to_docstore_id)

    def load(self):
        self.vectorstore = FAISS.load_local(
            self.store_path,
            self.embedding,
            allow_dangerous_deserialization=True,
        )
        with open(self.store_path / "source_to_docstore_id.pkl", "rb") as f:
            self.source_to_docstore_id = pickle.load(f)

    def save(self):
        self.vectorstore.save_local(self.store_path)
        with open(self.store_path / "source_to_docstore_id.pkl", "wb") as f:
            pickle.dump(self.source_to_docstore_id, f)

    def search(self, query: str, k: int = 1):
        return self.vectorstore.search(query, k=k)

    def delete(self, doc_id: str | None):
        if doc_id is not None:
            self.vectorstore.delete(doc_id)

    def delete_file(self, file_path: pathlib.Path):
        doc_id_to_delete = self.source_to_docstore_id.pop(file_path, None)
        self.delete(doc_id_to_delete)

    def delete_files(self, file_paths: list[pathlib.Path]):
        for fp in file_paths:
            self.delete_file(fp)

    def add_file(self, source: pathlib.Path):
        splitted_text = self.splitter.split_text(source.read_text())
        metadatas = [{"source": str(source)} for _ in splitted_text]
        text_ids = get_ids(splitted_text)
        self.vectorstore.add_texts(
            texts=splitted_text, metadatas=metadatas, ids=text_ids
        )
        for doc, id in zip(splitted_text, text_ids):
            if str(source) in self.source_to_docstore_id:
                self.source_to_docstore_id[str(source)].append(id)
            else:
                self.source_to_docstore_id[str(source)] = [id]

    def add_files(self, file_paths: list[pathlib.Path]):
        for fp in file_paths:
            self.add_file(fp)
