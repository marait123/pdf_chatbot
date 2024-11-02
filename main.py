from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, trim_messages, AIMessage, HumanMessage
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from dotenv import load_dotenv
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faiss_db import FAISSDatabase
from langchain.document_loaders import PyPDFLoader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
# create uploads directory if it does not exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")
MAX_CONTEXT_WINDOW = 512  # Define a constant for maximum context window


def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""

    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text


class Document(BaseModel):
    title: str
    path: str


uploaded_documents = []
messages = [
    #    SystemMessage("Welcome to the chat!"),
    #
]
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

faiss_db = FAISSDatabase()


# =================================================================================================
# APP global variables
# =================================================================================================
prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context and Chat History. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    Question: {input}
    Chat History:
    {chat_history}
    """)

# Import the necessary libraries

# Set up the API key for Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI client
genai.configure(api_key=api_key)

# Initialize the ChatGoogleGenerativeAI with the specified model and parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
document_chain = create_stuff_documents_chain(llm, prompt)

# =================================================================================================
# API Endpoints
# =================================================================================================


class PromptRequest(BaseModel):
    prompt: str


# return the main site files at ./frontend/index.html


@app.get("/", response_class=HTMLResponse)
async def main():
    with open("frontend/index.html") as f:
        return HTMLResponse(content=f.read())


# an endpoint where user can upload a file and this file should be save


@app.post("/documents/", response_model=Document)
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        print(f"Received file: {file.filename}")
        file_location = f"./uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Extract text from the uploaded PDF
        # extracted_text = extract_text_from_pdf(file_location)

        # Add the document to the uploaded_documents array
        document = Document(title=file.filename, path=file_location)
        uploaded_documents.append(document)
        loader = PyPDFLoader(file_location)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        documents = text_splitter.split_documents(docs)
        # Add the chunks to the FAISS database
        faiss_db.add_documents(documents)

        return document
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Failed to upload file"})


@app.get("/documents/")
def get_uploaded_documents():
    return {"documents": uploaded_documents}


@app.post("/chat/")
async def prompt(request: PromptRequest):
    """
    Handles the prompt request by retrieving relevant documents and generating a response.

    Args:
        request (PromptRequest): The request object containing the user's prompt.

    Returns:
        dict: A dictionary containing the AI's response to the user's prompt.
    """
    if not uploaded_documents:
        return JSONResponse(status_code=400, content={"message": "No documents uploaded. Please upload a document first."})

    try:
        retriever = faiss_db.db.as_retriever()
        user_query = request.prompt
        # Create retrieval chain by combining retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # form the chat history
        chat_history = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                chat_history += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                chat_history += f"AI: {message.content}\n"

        response = retrieval_chain.invoke({"input": user_query, "chat_history": chat_history})
        # now save the message to the messages array
        messages.append(HumanMessage(content=request.prompt))
        messages.append(AIMessage(content=response['answer']))

        return {"answer": response['answer']}
    except Exception as e:
        logger.error(f"Failed to process chat request: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Failed to process chat request"})

# Add exception handlers


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error occurred: {exc.errors()}")
    return JSONResponse(status_code=400, content={"message": "Validation error", "details": exc.errors()})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"An error occurred: {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
# command for running the server in development mode
# uvicorn main:app --reload
