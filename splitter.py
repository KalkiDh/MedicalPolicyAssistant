import os
import re
# MODIFIED: Updated import for modern LangChain versions (v0.1.0+)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_markdown(file_path: str) -> str:
    """
    Loads the content from a specified markdown file.

    Args:
        file_path (str): The absolute path to the markdown file.

    Returns:
        str: The content of the file as a string.
        
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found at: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_chunks(chunks: list, base_output_dir: str):
    """
    Saves a list of text chunks into a specified directory. Each chunk is saved as a separate file.

    Args:
        chunks (list): A list of strings, where each string is a text chunk.
        base_output_dir (str): The directory where the chunk files will be saved.
    """
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created chunks directory: {base_output_dir}")

    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(base_output_dir, f"chunk_{i+1}.md")
        with open(chunk_file_path, "w", encoding="utf-8") as f:
            f.write(chunk)
    
    print(f"✅ Successfully saved {len(chunks)} chunks to '{base_output_dir}'")

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits a given text into smaller chunks using a RecursiveCharacterTextSplitter.

    The splitter uses a list of markdown-specific separators to try and preserve
    the structure of the document (e.g., keeping paragraphs and list items intact).

    Args:
        text (str): The text content to be split.
        chunk_size (int): The maximum size (in characters) of each chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    print(f"Splitting text into chunks with size {chunk_size} and overlap {chunk_overlap}...")
    
    # Define separators that are common in markdown files
    # The splitter will try them in order from first to last.
    markdown_separators = [
        "\n## ",  # Split by section headers
        "\n### ", # Split by sub-section headers
        "\n\n",   # Split by paragraphs
        "\n",     # Split by new lines
        " ",      # Split by spaces
        "",       # As a final fallback
    ]

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=markdown_separators,
        length_function=len,
    )

    # Split the document
    chunks = text_splitter.split_text(text)
    return chunks

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # MODIFIED: Dynamically find the markdown file in the 'Exports' directory
        # instead of using a hardcoded name.
        exports_dir = "Exports"
        if not os.path.isdir(exports_dir):
            raise FileNotFoundError(f"The '{exports_dir}' directory was not found. Please run the document parser script first.")

        # Find any file in the directory that ends with .md
        markdown_files = [f for f in os.listdir(exports_dir) if f.endswith('.md')]

        if not markdown_files:
            raise FileNotFoundError(f"No markdown files found in '{exports_dir}'. Please ensure the document parser has run successfully.")

        # Use the first markdown file found in the directory
        input_md_filename = markdown_files[0]
        input_md_path = os.path.join(exports_dir, input_md_filename)
        print(f"Found and using input file: {input_md_path}")
        
        # 1. Load the parsed markdown content
        markdown_content = load_markdown(input_md_path)
        
        # 2. Split the content into chunks
        text_chunks = split_text_into_chunks(markdown_content)
        
        # 3. Define an output directory for the chunks
        # This creates a 'chunks' subdirectory inside the 'Exports' directory.
        chunks_output_dir = os.path.join(exports_dir, "chunks")
        
        # 4. Save the chunks to separate files
        save_chunks(text_chunks, chunks_output_dir)
        
        # Optional: Print the first chunk for verification
        if text_chunks:
            print("\n--- First Chunk for Verification ---")
            print(text_chunks[0])
            print("------------------------------------")

    except FileNotFoundError as fnf_error:
        print(f"\nError: {fnf_error}")
    except Exception as general_error:
        print(f"\nAn unexpected error occurred: {general_error}")
