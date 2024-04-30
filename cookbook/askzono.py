import os
import pathlib
from operator import itemgetter

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from pypdf import PdfReader

from aiasan import utils
from aiasan.vectorstore import VectorStore

# Setup
store_path = pathlib.Path(os.environ.get("OUTPUT_FOLDER")) / "store"
embedding = OllamaEmbeddings(model=os.environ.get("LOCAL_EMBED_MODEL"))

if "db" not in st.session_state:
    st.session_state.db = VectorStore.initialize_empty(store_path, embedding)

if "files" not in st.session_state:
    st.session_state.files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "k" not in st.session_state:
    st.session_state.k = 3

# Chains
llm = ChatOllama(temperature=0, model=os.environ.get("LOCAL_MODEL"))

## RAG
QUERY_INST = (
    "Given the above conversation history, generate a search query to look up in order to get information relevant to the last user question or query."
    "Only respond with the query, nothing else."
)
query_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="messages"), ("user", QUERY_INST)]
)
query_chain = RunnableBranch(
    (lambda x: len(x.get("messages", [])) == 1, lambda x: x["messages"][-1]["content"]),
    query_prompt | llm | StrOutputParser(),
)

retriever = st.session_state.db.vectorstore.as_retriever(
    search_kwargs={"k": st.session_state.k}
)
rag_chain = RunnablePassthrough.assign(query=query_chain).assign(
    context=itemgetter("query") | retriever
)
## Answer
SYSTEM_TEMPLATE = (
    "Answer the user's questions based on the below context."
    "Do not use sentences like 'based on the context' of 'depending on the knowledge'"
    "If the context doesn't contain any relevant information to the question, don't make something up and just say 'I don't know':"
    "<context>"
    "{context}"
    "</context>"
)
history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
answer_chain = create_stuff_documents_chain(llm, history_prompt)

# Streamlit App
with st.sidebar:
    st.session_state.k = st.number_input(
        "Number of documents used for the answer", 1, 20, 3
    )
    st.header(f"Add your documents!")
    uploaded_files = st.file_uploader(
        "Choose your `.pdf` or `.md` files",
        type=("pdf", "md"),
        accept_multiple_files=True,
    )
    if uploaded_files:
        for new_file in uploaded_files:
            file_key = new_file.name
            if file_key not in st.session_state.files:
                st.session_state.files.append(file_key)
                if new_file.type == "application/pdf":
                    splitter = RecursiveCharacterTextSplitter()
                    reader = PdfReader(new_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                elif new_file.type == "application/octet-stream":
                    splitter = MarkdownTextSplitter()
                    text = new_file.read().decode("utf-8")
                splits = splitter.split_text(text)
                st.session_state.db.add_texts(splits, [new_file.name for _ in splits])
                st.session_state.db.save()
        file = st.selectbox(
            "File", uploaded_files, index=None, format_func=lambda x: x.name
        )
        if file:
            utils.display(file)


st.title("Chat with your documents !")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status(f"Processing") as status:
            rag_result = rag_chain.invoke({"messages": st.session_state.messages})
            for doc in rag_result["context"]:
                st.write(f"- {doc.metadata["source"]}")
            status.update(
                label=f"Found {len(rag_result["context"])} documents about {rag_result["query"]}"
            )
        stream = answer_chain.stream(rag_result)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
