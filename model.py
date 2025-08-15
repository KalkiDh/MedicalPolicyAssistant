import os
import chromadb
from dotenv import load_dotenv

# Import the specific, modern classes from their dedicated packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Imports for the Azure LLM
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


def initialize_vector_store(persist_directory: str, embedding_function) -> Chroma:
    """
    Initializes and loads a persistent Chroma vector store from disk.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"The vector database directory was not found at: '{persist_directory}'. "
            "Please ensure you have run the embedding script first to create it."
        )
    print(f"Loading existing vector store from '{persist_directory}'...")
    vector_store = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_function
    )
    return vector_store

def perform_semantic_search(query: str, vector_store: Chroma, top_k: int = 5) -> list:
    """
    Performs a similarity search on the vector store to find relevant chunks.
    """
    print(f"\nPerforming semantic search for query: '{query}'")
    return vector_store.similarity_search(query, k=top_k)

def initialize_llm_client():
    """
    Initializes the Azure AI Inference client using environment variables.
    """
    load_dotenv() # Load environment variables from a .env file
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in environment variables. Please create a .env file and add it.")
    
    endpoint = "https://models.github.ai/inference"
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    print("LLM client initialized successfully.")
    return client

def create_prompt_with_context(query: str, relevant_chunks: list) -> str:
    """
    Combines the user's query with the content of relevant chunks to create a prompt for the LLM.
    """
    chunk_texts = [chunk.page_content for chunk in relevant_chunks]
    context = "\n\n---\n\n".join(chunk_texts)
    
    prompt = (
        f"Based on the following document excerpts, please answer the user's query.\n\n"
        f"--- CONTEXT ---\n{context}\n\n--- END CONTEXT ---\n\n"
        f"User Query: {query}"
    )
    return prompt

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # --- Setup Phase ---
        db_directory = "db"
        model_name = "openai/gpt-4o" # Using gpt-4o as specified in the logic

        # 1. Initialize the embedding model for searching
        print("Initializing embedding model (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # 2. Load the existing vector store
        vector_db = initialize_vector_store(db_directory, embeddings)

        # 3. Initialize the LLM client
        llm_client = initialize_llm_client()

        # 4. Setup conversation history with a system message
        system_prompt = SystemMessage(
            content="You are a helpful assistant that answers questions based *only* on the provided document context. "
                    "If the answer is not found in the context, say 'I cannot answer this based on the provided document.'"
        )
        conversation_history = [system_prompt]
        print("System prompt set. Ready to chat.")

        # --- Interactive Query Loop ---
        while True:
            user_query = input("\nEnter your query (or type 'quit' to exit): ").strip()
            if user_query.lower() == 'quit':
                print("Exiting program. Goodbye!")
                break
            if not user_query:
                continue

            # 1. Find relevant chunks from the document
            results = perform_semantic_search(user_query, vector_db)

            # 2. Create a detailed prompt for the LLM
            prompt_for_llm = create_prompt_with_context(user_query, results)
            
            # 3. Add the user's new message to the conversation history
            conversation_history.append(HumanMessage(content=prompt_for_llm))

            # 4. Get the LLM's response
            print("\nAI is thinking...")
            response = llm_client.complete(
                messages=conversation_history,
                model=model_name,
                temperature=0.7, # Lower temperature for more factual answers
            )
            ai_response_text = response.choices[0].message.content
            
            # 5. Add the AI's response to the history for future context
            conversation_history.append(AIMessage(content=ai_response_text))

            # 6. Display the result
            print(f"\nAI: {ai_response_text}")

    except FileNotFoundError as fnf_error:
        print(f"\nERROR: {fnf_error}")
    except ValueError as val_error:
        print(f"\nERROR: {val_error}")
    except Exception as general_error:
        print(f"\nAn unexpected error occurred: {general_error}")
