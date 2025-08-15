import os
# The chromadb import is not strictly necessary for this script to run,
# but it's good practice to acknowledge the underlying database technology.
import chromadb

# Import the specific, modern classes from their dedicated packages
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_vector_store(persist_directory: str, embedding_function) -> Chroma:
    """
    Initializes and loads a persistent Chroma vector store from disk.

    This function connects to the database directory where your embeddings are stored.

    Args:
        persist_directory (str): The directory where the ChromaDB is saved.
        embedding_function: The embedding function instance to use for querying.

    Returns:
        Chroma: The initialized Chroma vector store instance, ready for searching.
        
    Raises:
        FileNotFoundError: If the database directory does not exist.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"The vector database directory was not found at: '{persist_directory}'. "
            "Please ensure you have run the embedding script first to create it."
        )
    
    print(f"Loading existing vector store from '{persist_directory}'...")
    
    # Load the persisted database from disk
    vector_store = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_function
    )
    
    return vector_store

def perform_semantic_search(query: str, vector_store: Chroma, top_k: int = 5) -> list:
    """
    Performs a similarity search on the vector store to find relevant chunks.

    This function takes a text query, embeds it, and finds the most similar
    document chunks stored in the database.

    Args:
        query (str): The user's search query.
        vector_store (Chroma): The Chroma vector store instance to search in.
        top_k (int): The number of top similar documents to return.

    Returns:
        list: A list of LangChain Document objects representing the most similar chunks.
    """
    print(f"\nPerforming semantic search for query: '{query}'")
    
    # Use the similarity_search method to find the most relevant documents
    similar_docs = vector_store.similarity_search(query, k=top_k)
    return similar_docs

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Define the directory where the ChromaDB vector store is persisted.
        # This should be the same directory you used in your embedding script.
        db_directory = "db"

        # 1. Initialize the embedding model.
        # It's crucial to use the *exact same model* as you used for creating the embeddings.
        print("Initializing embedding model (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # 2. Load the existing vector store from the 'db' directory.
        vector_db = initialize_vector_store(db_directory, embeddings)

        # 3. Start an interactive loop to accept user queries.
        while True:
            user_query = input("\nEnter your query (or type 'quit' to exit): ").strip()
            if user_query.lower() == 'quit':
                print("Exiting program. Goodbye!")
                break
            if not user_query:
                continue

            # 4. Perform the semantic search with the user's query.
            results = perform_semantic_search(user_query, vector_db)

            # 5. Display the results in a readable format.
            if not results:
                print("No relevant chunks found for your query.")
            else:
                print(f"\nFound {len(results)} relevant chunks:")
                print("-" * 70)
                for i, doc in enumerate(results):
                    # Each 'doc' is a LangChain Document object containing the text
                    print(f"--- Chunk {i+1} ---\n")
                    print(doc.page_content)
                    print("-" * 70)

    except FileNotFoundError as fnf_error:
        print(f"\nERROR: {fnf_error}")
    except Exception as general_error:
        print(f"\nAn unexpected error occurred: {general_error}")
