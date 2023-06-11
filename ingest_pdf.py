


"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()

def ingest_docs():
    """Get documents from web pages."""
    loader = PyPDFLoader("data/iv3_um.pdf")
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(pages, embeddings)

    # Save vectorstore
    with open("iv_vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()

