import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

def run_llm(query: str):
    
    # Connecting to Pinecone
    pinecone.init(api_key=os.environ.get("PINECONE_SECRET_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"))
 
    embeddings = OpenAIEmbeddings()
    
    # Connecting to Pinecone index
    doc_search = Pinecone.from_existing_index(

        index_name="greta",
        embedding=embeddings
    )
    
    # Connecting chat model API key and giving settings for LLM 
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

    chat_history = []

    # Message for user
    print("\nHi there! I'm Greta, your climate change assistant :) \n\nI'm here to answer your questions. \nWhen you had enough of me, just type 'bye' and I will be gone. \n\nWhat would you like to know?\n")

    while True:
        user_input = input("\nYou: ")

        if user_input == "bye":
            break

        if not user_input:
            print("I'm sure there is something you would like to know :)")
            continue
        
        response = qa({"question": user_input, "chat_history": chat_history})

        print(f"\nResponse: {response.get('answer')}")

        chat_history.append((user_input, response))

    

@dataclass

class PineconeCredentials:

    api_key: str
    index_name: str
    environment_region: str

 

def init_pinecone():

    api_key = os.environ.get("PINECONE_SECRET_KEY")
    index_name = "greta"
    environment = os.environ.get("PINECONE_ENVIRONMENT_REGION")
    

    if api_key is None:

        msg = "PINECONE_API_KEY environment variable is not set."
        raise ValueError(msg)

    
    if index_name is None:

        msg = "PINECONE_INDEX_NAME environment variable is not set."
        raise ValueError(msg)
   

    if environment is None:

        msg = "PINECONE_ENVIRONMENT_REGION environment variable is not set."
        raise ValueError(msg)

    
    return PineconeCredentials(

        api_key=api_key,
        index_name=index_name,
        environment_region=environment
    )


if __name__ == "__main__":

    pinecone_env = init_pinecone()
    run_llm(pinecone_env)
