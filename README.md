# RAG-Enabled Local LLM Assistant

## About the Project

This mini-project demonstrates how to enhance large language model (LLM) responses using **Retrieval-Augmented Generation (RAG)**.  
RAG allows the model to retrieve relevant documents from a vector store and incorporate them into its responses, enabling more accurate and context-aware answers.

The project uses the following technologies:

- **Ollama**: For running local LLMs efficiently
- **ChromaDB**: As a persistent vector store for document embeddings
- **LangChain**: For document processing and text splitting
- **Sentence Transformers**: For embedding generation

The project structure follows a modular design covering:

1. Data ingestion
2. Document chunking
3. Embedding generation
4. Vector storage
5. RAG-based retrieval
6. LLM integration for chat interaction

---

## Setup

### Prerequisites

1. **Docker Desktop** installed and running
2. **Ollama** running on Docker container
3. **Git Bash**
4. **.env** file with required environemnt variables. Refer to `.env.example` for a sample

### Project Setup

Run the setup script from the project root:

```bash
./setup.sh
```

This script will:

1. Install required Python dependencies
2. Set up environment variables
3. Prepare the project structure

## Running the Chat Application

After setting up the environment:

Start the RAG assistant CLI:

```bash
python -m main
```

If you want to embed new or updated documents:

```bash
python -m main --reindex
```

Follow the prompts:

You will see a welcome message from the assistant.

Type a question and press Enter.

To exit the chat, type:

`exit` or `end`
