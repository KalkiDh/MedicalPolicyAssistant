import os
from urllib.parse import urlparse
from PyPDF2 import PdfReader

def parse_local_document(local_path: str, output_dir: str = "Exports") -> str:
    """
    Parses a local PDF document using PyPDF2 and extracts its text content into a Markdown file.

    Args:
        local_path (str): The absolute path to the local PDF document file.
        output_dir (str): The directory where the extracted Markdown file will be saved.

    Returns:
        str: The path to the generated Markdown file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the provided file is not a PDF.
        Exception: If parsing fails.
    """
    # --- 1. Validate Input and Set Up Paths ---
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"No file found at the specified path: {local_path}")

    if not local_path.lower().endswith('.pdf'):
        raise ValueError("This script currently only supports PDF files.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Create a clean filename for the output markdown file
    base_name = os.path.basename(local_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_md_path = os.path.join(output_dir, f"{file_name_without_ext}.md")

    print(f"Processing document: {local_path}")

    # --- 2. Attempt Parsing with PyPDF2 for PDFs ---
    try:
        print("Attempting text extraction with PyPDF2...")
        reader = PdfReader(local_path)
        text_content = ""
        for page in reader.pages:
            # Extract text and handle pages with no text
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n\n"
        
        # If text was successfully extracted, save it and return
        if text_content.strip():
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✅ Successfully extracted text with PyPDF2. Saved to: {output_md_path}")
            return output_md_path
        else:
            # Raise an error if no text could be extracted
            raise Exception("PyPDF2 parsing resulted in no extractable text.")
            
    except Exception as e:
        # If parsing fails, raise an exception with the details
        raise Exception(f"Failed to parse PDF document. Error: {str(e)}")

# --- Example Usage ---
if __name__ == "__main__":
    # To test this script, replace this path with the absolute path to a document on your system.
    # For example: "C:\\Users\\YourUser\\Documents\\sample.pdf"
    try:
        # Prompt the user to enter the full path to their document
        document_path_input = input("Enter the full, absolute path to your PDF document: ").strip()
        
        if document_path_input:
            # Call the main function to parse the document
            markdown_file_path = parse_local_document(document_path_input)
            print(f"\nParsing complete. You can find the output here: {markdown_file_path}")
        else:
            print("No path was provided. Please run the script again and enter a valid file path.")
            
    except (FileNotFoundError, ValueError) as known_error:
        print(f"\nError: {known_error}")
    except Exception as general_error:
        print(f"\nAn unexpected error occurred: {general_error}")
