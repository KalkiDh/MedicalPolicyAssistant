import os
import shutil
from dotenv import load_dotenv

# --- 1. Import Core Logic from Your Existing Modules ---
# These imports assume your other .py files are in the same directory
# and contain the functions with the specified names.
from document_parser import parse_local_document
from splitter import split_text_into_chunks, load_markdown
# Import embeddings-related functions from embeddings.py
from embeddings import create_and_store_embeddings

# LangChain and related imports
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Azure LLM imports
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# --- LLM and Session Management ---
# This part can also be moved to its own 'llm_handler.py' for even cleaner code
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Gets the chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def initialize_llm():
    """Initializes and returns the Azure AI client."""
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in environment variables.")
    
    return ChatCompletionsClient(
        endpoint="https://models.github.ai/inference",
        credential=AzureKeyCredential(token),
    )

# --- Main Application ---
if __name__ == "__main__":
    try:
        # --- 2. Define Wrappers for LangChain Compatibility ---
        # These wrappers adapt your existing functions to the input/output
        # dictionary format that LangChain chains expect.

        def parse_wrapper(inputs: dict) -> dict:
            """Runnable wrapper for the document parser."""
            doc_path = inputs["doc_path"]
            md_path = parse_local_document(doc_path)
            return {"md_path": md_path}

        def split_wrapper(inputs: dict) -> dict:
            """Runnable wrapper for the text splitter."""
            md_path = inputs["md_path"]
            content = load_markdown(md_path)
            chunks = split_text_into_chunks(content)
            return {"chunks": chunks}

        def embed_wrapper(inputs: dict) -> dict:
            """Runnable wrapper for the embedding creator."""
            db_path = "db"
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            vector_store = create_and_store_embeddings(inputs["chunks"], db_path)
            return {"retriever": vector_store.as_retriever(search_kwargs={'k': 5})}

        # --- 3. Build the Document Processing Chain ---
        # This chain pipes the output of one function to the input of the next.
        processing_chain = (
            RunnableLambda(parse_wrapper)
            | RunnableLambda(split_wrapper)
            | RunnableLambda(embed_wrapper)
        )

        # --- 4. Execute the Processing Chain ---
        doc_path = input("Enter the full path to your PDF document: ").strip()
        if not doc_path:
            raise ValueError("Document path cannot be empty.")
            
        print("\n--- Starting Document Processing Pipeline ---")
        result = processing_chain.invoke({"doc_path": doc_path})
        retriever = result["retriever"]
        print("--- Document Processing Complete ---\n")

        # --- 5. Set up the Conversational RAG Chain ---
        llm_client = initialize_llm()
        model_name = "openai/gpt-4o"

        # Custom wrapper to make the LLM client LangChain-compatible
        def llm_invoke(prompt):
            if isinstance(prompt, dict):
                messages = prompt["messages"]
            else:
                messages = prompt.to_messages()
            
            # Convert messages to format expected by Azure API
            formatted_messages = []
            for msg in messages:
                # Get the role and content
                if hasattr(msg, 'type'):
                    role = msg.type
                    content = msg.content
                else:
                    role = msg["role"]
                    content = msg["content"]
                
                # Map roles to valid Azure API roles
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            
            response = llm_client.complete(messages=formatted_messages, model=model_name)
            return response.choices[0].message.content

        # Chain to rephrase follow-up questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and a follow up question, rephrase the question to be a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            RunnableLambda(llm_invoke), retriever, contextualize_q_prompt
        )

        # Chain to answer questions based on context
        qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(RunnableLambda(llm_invoke), qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # --- 6. Start the Interactive Chat ---
        session_id = "rag_chat_session"
        print("--- You can now ask questions about the document. ---")
        while True:
            user_query = input("\nEnter your query (or 'quit' to exit): ").strip()
            if user_query.lower() == 'quit':
                break
            
            chat_history = get_session_history(session_id)
            
            # Convert chat history to a format that can be serialized
            formatted_history = []
            for msg in chat_history.messages:
                role = msg.type if hasattr(msg, 'type') else "user" if "human" in str(msg.type).lower() else "assistant"
                # Map roles to valid Azure API roles
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                formatted_history.append({
                    "role": role,
                    "content": msg.content
                })
            
            response = rag_chain.invoke(
                {"input": user_query, "chat_history": formatted_history}
            )
            
            # Manually update chat history
            chat_history.add_user_message(user_query)
            chat_history.add_ai_message(response["answer"])
            
            print(f"\nAI: {response['answer']}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up generated directories
        if os.path.exists("Exports"):
            shutil.rmtree("Exports")
        # Give ChromaDB a chance to close its files
        import time
        time.sleep(1)
        if os.path.exists("db"):
            try:
                shutil.rmtree("db")
            except PermissionError:
                print("\nNote: Could not remove the db directory. It may be in use.")
        print("\nCleanup complete. Goodbye!")
