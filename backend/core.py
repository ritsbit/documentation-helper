import os
from dotenv import load_dotenv

load_dotenv()

from typing import Any, List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone as PineconeLangChain

# Pinecone client
from pinecone import Pinecone

# Create the Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

from consts import INDEX_NAME


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # instantiate the LLM
    chatllm = ChatOpenAI(temperature=0, verbose=True)

    qa = RetrievalQA.from_chain_type(
        llm=chatllm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # instantiate the LLM
    chatllm = ChatOpenAI(temperature=0, verbose=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chatllm, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(
        run_llm(
            query="Agent getting error while logging in Softphone. What action they should take?"
        )
    )
