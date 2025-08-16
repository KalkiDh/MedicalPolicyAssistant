import os
import shutil
import time
import tempfile
import logging
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Broad access for testing; replace with ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
llm_client = None
vector_store = None
histories: Dict[str, BaseChatMessageHistory] = {}

# Load environment variables
load_dotenv()

# --- Core Logic Functions ---

def parse_local_document(local_path: str, output_dir: str = "Exports") -> str:
    logger.info(f"Parsing document at: {local_path}")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"No file found at the specified path: {local_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.basename(local_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_md_path = os.path.join(output_dir, f"{file_name_without_ext}.md")

    try:
        reader = PdfReader(local_path)
        text_content = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n\n"
        
        if text_content.strip():
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logger.info(f"Document parsed successfully, saved to: {output_md_path}")
            return output_md_path
        else:
            raise Exception("PyPDF2 parsing resulted in no extractable text.")
            
    except Exception as e:
        logger.error(f"Failed to parse PDF document: {str(e)}")
        raise Exception(f"Failed to parse PDF document. Error: {str(e)}")

def load_markdown(file_path: str) -> str:
    logger.info(f"Loading markdown from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    logger.info("Splitting text into chunks")
    markdown_separators = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=markdown_separators,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def add_document_to_store(chunks: list, doc_id: str):
    global vector_store
    logger.info(f"Adding document {doc_id} with {len(chunks)} chunks to vector store")
    if not chunks:
        raise ValueError("No chunks provided to add.")
    # Delete existing chunks for this doc_id
    vector_store.delete(where={"doc_id": doc_id})
    # Add new chunks
    vector_store.add_texts(
        texts=chunks,
        metadatas=[{"doc_id": doc_id} for _ in chunks]
    )
    logger.info(f"Document {doc_id} added to vector store")

def get_session_history(doc_id: str) -> BaseChatMessageHistory:
    logger.info(f"Retrieving session history for doc_id: {doc_id}")
    if doc_id not in histories:
        histories[doc_id] = ChatMessageHistory()
    return histories[doc_id]

def llm_invoke(input_data):
    logger.info("Invoking LLM")
    if isinstance(input_data, dict) and "messages" in input_data:
        messages = input_data["messages"]
    else:
        try:
            messages = input_data.to_messages()
        except:
            messages = input_data

    formatted_messages = []
    for msg in messages:
        if hasattr(msg, 'type'):
            role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
            content = msg.content
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
        formatted_messages.append({"role": role, "content": content})
    
    response = llm_client.complete(
        messages=formatted_messages,
        model="openai/gpt-4o",
        temperature=0.7,
    )
    logger.info("LLM invocation successful")
    return response.choices[0].message.content

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    global llm_client
    global vector_store
    logger.info("Starting up FastAPI application")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN not found in environment variables")
        raise ValueError("GITHUB_TOKEN not found in environment variables.")
    
    llm_client = ChatCompletionsClient(
        endpoint="https://models.github.ai/inference",
        credential=AzureKeyCredential(token),
    )
    logger.info("LLM client initialized")

    # Clean up db on startup to ensure no permanent storage
    if os.path.exists("db"):
        logger.info("Cleaning up db directory on startup")
        shutil.rmtree("db")

    # Initialize vector store
    embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = Chroma(persist_directory="db", embedding_function=embedding_function)
    logger.info("Vector store initialized")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down FastAPI application")
    if os.path.exists("Exports"):
        shutil.rmtree("Exports")
    time.sleep(1)
    if os.path.exists("db"):
        try:
            shutil.rmtree("db")
        except PermissionError:
            logger.warning("PermissionError when deleting db directory")

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str
    doc_id: str
    session_id: str = "default"

class HistoryRequest(BaseModel):
    doc_id: str

# --- Endpoints ---

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        doc_id = file.filename
        logger.info(f"Uploading file: {file.filename} as doc_id: {doc_id}")
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type: not a PDF")
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Clean up Exports directory
        if os.path.exists("Exports"):
            logger.info("Cleaning up Exports directory")
            shutil.rmtree("Exports")

        # Save uploaded file to temp path with .pdf extension
        logger.info("Saving uploaded file to temporary path")
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            doc_path = temp_file.name
            logger.info(f"Temporary file saved at: {doc_path}")

        # Process document
        md_path = parse_local_document(doc_path)
        content = load_markdown(md_path)
        chunks = split_text_into_chunks(content)
        add_document_to_store(chunks, doc_id)

        # Clean up temp file and Exports
        logger.info(f"Cleaning up temporary file: {doc_path}")
        os.unlink(doc_path)
        if os.path.exists("Exports"):
            shutil.rmtree("Exports")

        # Initialize history if not exists
        if doc_id not in histories:
            histories[doc_id] = ChatMessageHistory()
        logger.info("Document upload and processing successful")

        return {"message": "Document uploaded and processed successfully.", "doc_id": doc_id}

    except ValueError as ve:
        logger.error(f"ValueError in upload: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_document(q: QueryRequest):
    logger.info(f"Processing query for doc_id: {q.doc_id}, query: {q.query}")
    if vector_store is None:
        logger.error("Vector store not initialized")
        raise HTTPException(status_code=500, detail="Vector store not initialized.")

    try:
        chat_history = get_session_history(q.doc_id)
        formatted_history = [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in chat_history.messages
        ]

        # Create retriever with filter for doc_id
        retriever = vector_store.as_retriever(search_kwargs={'k': 5, 'filter': {'doc_id': q.doc_id}})

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and a follow up question, rephrase the question to be a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            RunnableLambda(llm_invoke), retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(RunnableLambda(llm_invoke), qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = rag_chain.invoke(
            {"input": q.query, "chat_history": formatted_history}
        )

        chat_history.add_user_message(q.query)
        chat_history.add_ai_message(response["answer"])
        logger.info("Query processed successfully")

        return {"answer": response["answer"]}

    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.post("/history")
async def get_history(h: HistoryRequest):
    logger.info(f"Fetching history for doc_id: {h.doc_id}")
    try:
        chat_history = get_session_history(h.doc_id)
        history = [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in chat_history.messages
        ]
        logger.info(f"History retrieved for doc_id: {h.doc_id}, {len(history)} messages")
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")