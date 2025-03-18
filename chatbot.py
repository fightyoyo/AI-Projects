from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os
import shutil
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf-query-app")

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.critical("GROQ_API_KEY environment variable is not set. Please configure it in your .env file.")
    raise ValueError("GROQ_API_KEY environment variable is not set. Please configure it in your .env file.")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. Be as specific as possible.
    <context>
    {context}
    </context>
    <context>
    Question: {input}
    </context>
    """
)

# Global variable for vector store
vector_store = None

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile]):
    """
    Endpoint to upload and process multiple PDF files.
    """
    global vector_store
    try:
        if not files:
            logger.error("No files were uploaded.")
            raise HTTPException(status_code=400, detail="No files uploaded.")

        all_text_documents = []

        for file in files:
            if file.content_type != "application/pdfs":
                logger.error(f"Invalid file type: {file.filename} ({file.content_type})")
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a valid PDF file.")

            temp_pdf_path = f"temp_{file.filename}"
            try:
                logger.debug(f"Saving uploaded file {file.filename} temporarily.")
                with open(temp_pdf_path, "wb") as temp_file:
                    shutil.copyfileobj(file.file, temp_file)

                # Extract content from the uploaded PDF
                text_documents = extract_pdf_content(temp_pdf_path)
                all_text_documents.extend(text_documents)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        if not all_text_documents:
            logger.error("No content extracted from the uploaded PDFs.")
            raise HTTPException(status_code=400, detail="No content extracted from the uploaded PDFs.")

        # Create a vector store from all the combined text documents
        vector_store = create_vector_db_from_text(all_text_documents)

        logger.debug("Vector store created successfully.")
        return JSONResponse(content={"message": "PDFs processed successfully.", "total_chunks": vector_store.index.ntotal})

    except Exception as e:
        logger.exception("Error processing PDFs.")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")


@app.post("/query-pdf/")
async def query_pdf(question: str):
    """
    Endpoint to query the processed PDF content.
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=400, detail="No PDF content has been processed yet.")

        answer = get_answer_from_query(question, vector_store)
        return JSONResponse(content={"question": question, "answer": answer})

    except Exception as e:
        logger.exception("Error retrieving answer.")
        raise HTTPException(status_code=500, detail=f"Error retrieving answer: {str(e)}")


def extract_pdf_content(pdf_file_path):
    """
    Extracts content from a PDF file.
    """
    loader = PyPDFLoader(pdf_file_path)
    text_documents = loader.load()
    logger.debug(f"Extracted {len(text_documents)} documents from {pdf_file_path}.")
    return text_documents


def create_vector_db_from_text(text_documents):
    """
    Creates a vector store from the extracted text.
    """
    global vector_store

    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    document_chunks = text_splitter.split_documents(text_documents)

    logger.debug(f"Number of document chunks: {len(document_chunks)}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},  # Change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create a vector store
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    return vector_store


def get_answer_from_query(query, vector_store):
    """
    Answers a query using the vector store.
    """
    if not vector_store:
        logger.error("Vector store is empty. Ensure the PDFs are properly processed.")
        return "Vector store is empty. Ensure the PDFs are properly processed."

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    logger.debug(f"Retrieving answer for query: {query}")
    response = retrieval_chain.invoke({"input": query})

    return response.get("answer", "No relevant answer found in the context.")
