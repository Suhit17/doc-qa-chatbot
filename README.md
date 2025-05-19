# Document QA Chatbot

A Streamlit-based chatbot that can answer questions about your documents using OpenAI's language models and LangChain.

## Features

- Upload and process multiple documents (PDF, TXT, MD)
- Interactive chat interface
- Support for different OpenAI models (GPT-4, GPT-3.5-turbo)
- Document chunking and vector storage using FAISS
- Conversation memory for context-aware responses

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get an OpenAI API key from [OpenAI's platform](https://platform.openai.com/api-keys)

## Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Enter your OpenAI API key in the sidebar
4. Upload your documents and start asking questions!

## Usage

1. Enter your OpenAI API key in the sidebar
2. Select your preferred OpenAI model
3. Upload one or more documents (PDF, TXT, or MD files)
4. Click "Process Documents" to analyze the content
5. Start asking questions about your documents in the chat interface

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API access

## Note

This application uses OpenAI's API, which may incur costs depending on your usage. Please check OpenAI's pricing page for more information. 