import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# This is the LangChain version of Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# This is for Pinecone client
from pinecone import Pinecone

from consts import INDEX_NAME

# Create the Pinecone client
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    # Load docs

    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    # loader = TextLoader(file_path="ccaas-docs-word/JIRA Template for Softphone.txt")

    loader = DirectoryLoader(
        path="ccaas-docs-word", glob="**/*.txt", loader_cls=TextLoader
    )

    print("Loading documents from langchain docs site...")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    print("Splitting documents from langchain docs...")
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    embeddings = OpenAIEmbeddings()
    print("Adding embeddings to Pinecone DB...")
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )
    print("***** Added documents to PineconeVectorStore ***** ")


if __name__ == "__main__":
    print("Starting ingestion")
    ingest_docs()
