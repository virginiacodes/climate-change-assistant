from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv

directory = "C:\\Users\\VirginieHelmreich\\Documents\\xander\\climate_assistant\\chatbot\\assets\\Independent-Assessment-of-UK-Climate-Risk-Advice-to-Govt-for-CCRA3-CCC.pdf"

load_dotenv()

# Load PDFs from local folder to Pinecone.
def ingest_text():

    pinecone.init(api_key=os.environ.get("PINECONE_SECRET_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"))

    # Pinecone index name, found on Pinecone account.
    index_name = "greta"

    loader = PyPDFLoader(directory)
    raw_text = loader.load()
  
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    # Splitting text in chunks of 200 characters
    documents = text_splitter.split_documents(raw_text)

    #Embedding raw text and sending it to Pinecone
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get ("OPENAI_API_KEY"))
    index = Pinecone.from_documents(documents=documents, embedding=embeddings, index_name=index_name)

if __name__ == "__main__":
    ingest_text()   