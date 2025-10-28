from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()
EMBEDDING_MODEL = config['models']['embedding_model']

DEFAULT_IGNORE_PATTERNS = [
    "**/.git/*",
    "**/.venv/*",
    "**/venv/*",
    "**/.idea/*",
    "**/__pycache__/*",
    "**/node_modules/*",
    "**/*.pyc",
    "**/.DS_Store",
]

def load_and_embed_code(code_directory: str):
    """
    Loads Python files from a directory, splits them, and embeds them
    into a Chroma vector store.
    """
    print(f"[RAG] Loading code from: {code_directory}")


    loader = DirectoryLoader(
        code_directory,
        glob="**/*.py",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        exclude=DEFAULT_IGNORE_PATTERNS
    )
    documents = loader.load()

    if not documents:
        print("[RAG] No .py files found.")
        return None


    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python", chunk_size=1000, chunk_overlap=100
    )
    chunks = python_splitter.split_documents(documents)
    print(f"[RAG] Split {len(documents)} files into {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(chunks, embeddings)

    print("[RAG] Code embedding complete.")
    return vector_store.as_retriever(search_kwargs={"k": 5})