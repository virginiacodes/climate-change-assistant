import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from dataclasses import dataclass

if os.path.exists("env.py"):
    import env


def run_llm(query: str):

    pinecone.init(api_key=os.environ.get("PINECONE_SECRET_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"))
 
    embeddings = OpenAIEmbeddings()

    doc_search = Pinecone.from_existing_index(

        index_name="greta",
        embedding=embeddings
    )

    chat_model = ChatOpenAI(

        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=1,
        verbose=True
    )


    qa = ConversationalRetrievalChain.from_llm(

        llm=chat_model,

        retriever=doc_search.as_retriever()

    )


