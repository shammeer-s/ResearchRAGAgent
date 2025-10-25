from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# This is the embedding model we downloaded
EMBEDDING_MODEL = "nomic-embed-text"

def load_and_embed_code(code_directory: str):
    """
    Loads Python files from a directory, splits them, and embeds them
    into a Chroma vector store.
    """
    print(f"[RAG] Loading code from: {code_directory}")

    # 1. Load
    loader = DirectoryLoader(
        code_directory,
        glob="**/*.py", # Load only .py files
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    if not documents:
        print("[RAG] No .py files found.")
        return None

    # 2. Split (using Python-aware splitter)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python", chunk_size=1000, chunk_overlap=100
    )
    chunks = python_splitter.split_documents(documents)
    print(f"[RAG] Split {len(documents)} files into {len(chunks)} chunks.")

    # 3. Embed and Store
    # This uses the 'nomic-embed-text' model you pulled
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Using an in-memory vector store
    vector_store = Chroma.from_documents(chunks, embeddings)

    print("[RAG] Code embedding complete.")
    return vector_store.as_retriever(search_kwargs={"k": 3}) # Return top 3 results