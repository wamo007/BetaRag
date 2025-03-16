# RAG Chatbot with Groq

This is a Retrieval-Augmented Generation (RAG) chatbot implementation using Groq's LLM API, ChromaDB for vector storage, and local embeddings using Transformers.js.

## Features

- Uses Groq's Mixtral 8x7B model for chat completions
- Local vector storage using ChromaDB
- Local embeddings using Transformers.js (all-MiniLM-L6-v2 model)
- Simple web interface for chatting
- Conversation memory with semantic search

## Prerequisites

- Node.js
- npm
- Python11 for ChromaDB
- A Groq API key

## Setup the app

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create an `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   PORT=3000
   ```

## Setup the database for semantic vectors

1. In the root directory activate virtual python environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install and run ChromaDB, directing it to db folder:
   ```
   pip install chromadb
   chroma run --path db
   ```

## Running the Application

1. Start the server:
   ```bash
   npm run dev
   ```
2. Open your browser and navigate to `http://localhost:3000`

## How it Works

1. When a user sends a message, the system:
   - Generates embeddings for the message using Transformers.js
   - Searches the vector database (ChromaDB) for relevant past messages
   - Combines the relevant history with the current message
   - Sends the combined context to Groq's API
   - Stores both the user message and AI response in the vector database

2. The vector database maintains a history of all conversations, allowing the chatbot to reference past interactions when relevant to the current conversation.

## Technologies Used

- Groq API - Large Language Model
- ChromaDB - Vector Database
- Transformers.js - Local Embeddings
- Express.js - Web Server
- Node.js - Runtime Environment 
- Python - ChromaDB Runtime Environment