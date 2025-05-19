import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set page configuration
st.set_page_config(page_title="Document QA Chatbot", page_icon="📚", layout="wide")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Set up the app UI
st.title("📚 Document QA Chatbot")

# Create sidebar for API key and document upload
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Model selection
    model_options = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
    selected_model = st.selectbox("Select Model:", model_options)
    
    # Document upload section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    # Process button
    if uploaded_files and api_key:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Load documents
                documents = []
                
                for file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                        temp_file.write(file.getvalue())
                        temp_path = temp_file.name
                    
                    # Use appropriate loader based on file type
                    if file.name.endswith('.pdf'):
                        loader = PyPDFLoader(temp_path)
                    elif file.name.endswith('.txt'):
                        loader = TextLoader(temp_path)
                    elif file.name.endswith('.md'):
                        loader = UnstructuredMarkdownLoader(temp_path)
                    
                    # Load the document
                    documents.extend(loader.load())
                    # Clean up the temp file
                    os.unlink(temp_path)
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings and vector store
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store
                
                # Set up conversation chain
                llm = ChatOpenAI(model_name=selected_model, temperature=0.0)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                    memory=memory,
                    verbose=True
                )
                
                st.session_state.conversation = conversation_chain
                st.session_state.processing_complete = True
                st.success(f"Processed {len(documents)} documents into {len(chunks)} chunks!")

# Main chat interface
if api_key:
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_question := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Check if documents have been processed
        if st.session_state.conversation is not None:
            # Display assistant message with a spinner while generating
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response from conversation chain
                    response = st.session_state.conversation({"question": user_question})
                    answer = response["answer"]
                    
                    # Display the answer
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            # Display message if documents haven't been processed yet
            with st.chat_message("assistant"):
                st.write("Please upload and process documents first before asking questions.")
                st.session_state.chat_history.append({"role": "assistant", "content": "Please upload and process documents first before asking questions."})
else:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")

# Add some helpful information at the bottom
st.divider()
st.markdown("""
### How to use this app:
1. Enter your OpenAI API key in the sidebar
2. Upload documents (PDF, TXT, MD files)
3. Click "Process Documents"
4. Ask questions about your documents in the chat input
""") 