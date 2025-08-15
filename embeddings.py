import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_chunks_from_directory(chunks_dir: str) -> list:
    """
    Loads the content of all markdown chunk files from a specified directory.

    Args:
        chunks_dir (str): The path to the directory containing the chunk files.

    Returns:
        list: A list of strings, where each string is the content of a chunk file.
        
    Raises:
        FileNotFoundError: If the chunks directory does not exist.
    """
    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(f"The specified chunks directory was not found: {chunks_dir}")

    chunk_texts = []
    print(f"Loading chunks from '{chunks_dir}'...")
    # Sort the files to ensure they are loaded in a consistent order
    for filename in sorted(os.listdir(chunks_dir)):
        if filename.endswith(".md"):
            file_path = os.path.join(chunks_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_texts.append(f.read())
    
    print(f"Found and loaded {len(chunk_texts)} chunks.")
    return chunk_texts

def create_and_store_embeddings(chunks: list, persist_directory: str):
    """
    Creates embeddings for a list of text chunks and stores them in a persistent ChromaDB.

    Args:
        chunks (list): The list of text chunks to embed.
        persist_directory (str): The directory on disk where the ChromaDB will be saved.
    """
    if not chunks:
        print("No chunks provided to embed. Exiting.")
        return

    print("Initializing embedding model (this may download the model on first run)...")
    # Initialize the embedding model from HuggingFace
    embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    print(f"Creating and persisting vector store at '{persist_directory}'...")
    # Create the Chroma vector store from the text chunks
    # This will handle the embedding process and save the results to the specified directory.
    vector_store = Chroma.from_texts(
        texts=chunks, 
        embedding=embedding_function, 
        persist_directory=persist_directory
    )

    # Persist the vector store to disk
    vector_store.persist()
    
    print(f"✅ Successfully created and stored {len(chunks)} embeddings in ChromaDB.")
    
    return vector_store

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Define the directory where the text chunks are stored
        chunks_input_dir = os.path.join("Exports", "chunks")
        
        # Define the directory where the ChromaDB vector store will be persisted
        db_output_dir = "db"

        # 1. Load the text chunks from their files
        text_chunks = load_chunks_from_directory(chunks_input_dir)
        
        # 2. Create embeddings and store them in ChromaDB
        create_and_store_embeddings(text_chunks, db_output_dir)

    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
    except Exception as general_error:
        print(f"\nAn unexpected error occurred: {general_error}")
