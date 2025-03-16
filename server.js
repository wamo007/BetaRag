require('dotenv').config();
const express = require('express');
const path = require('path');
const Groq = require('groq-sdk');
const { ChromaClient } = require('chromadb');
const { pipeline } = require('@xenova/transformers');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Initialize Groq client
const groq = new Groq(process.env.GROQ_API_KEY);

// Initialize ChromaDB client
const chromaClient = new ChromaClient({
  path: "http://localhost:8000"  // Default ChromaDB server URL
});
let collection;

// Initialize the embedding model
let embedder;

// Initialize the chat history collection
async function initializeVectorDB() {
  try {
    // Get or create the collection
    collection = await chromaClient.getCollection({
      name: "chat_history"
    });
    
    if (!collection) {
      collection = await chromaClient.createCollection({
        name: "chat_history",
        metadata: { "description": "Store chat history embeddings" }
      });
    }
    
    console.log('Vector DB initialized successfully');
  } catch (error) {
    console.error('Error initializing vector DB:', error);
    throw error;
  }
}

// Initialize the embedding model
async function initializeEmbedder() {
  try {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Embedding model initialized successfully');
  } catch (error) {
    console.error('Error initializing embedding model:', error);
    throw error;
  }
}

// Generate embeddings for text
async function generateEmbeddings(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Store chat message in vector DB
async function storeChatMessage(message, role) {
  try {
    const embedding = await generateEmbeddings(message);
    await collection.add({
      ids: [Date.now().toString()],
      embeddings: [embedding],
      metadatas: [{ role, timestamp: new Date().toISOString() }],
      documents: [message]
    });
  } catch (error) {
    console.error('Error storing chat message:', error);
  }
}

// Retrieve relevant chat history
async function getRelevantHistory(currentMessage, limit = 5) {
  try {
    const queryEmbedding = await generateEmbeddings(currentMessage);
    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: limit
    });
    
    // Check if there are any results
    if (!results.documents || !results.documents.length || !results.documents[0].length) {
      return [];
    }

    // Map the results to the expected format
    return results.documents[0].map((doc, index) => ({
      content: doc,
      role: results.metadatas[0][index].role
    }));
  } catch (error) {
    console.error('Error retrieving chat history:', error);
    return [];
  }
}

// Chat endpoint
app.post('/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Get relevant chat history
    const relevantHistory = await getRelevantHistory(message);
    
    // Construct the conversation for Groq
    const conversation = [
      ...relevantHistory.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      { role: 'user', content: message }
    ];

    // Get response from Groq
    const completion = await groq.chat.completions.create({
      messages: conversation,
      model: 'mixtral-8x7b-32768',
      temperature: 0.7,
      max_tokens: 1024,
    });

    const assistantResponse = completion.choices[0].message.content;

    // Store both user message and assistant response
    await storeChatMessage(message, 'user');
    await storeChatMessage(assistantResponse, 'assistant');

    res.json({ response: assistantResponse });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Initialize everything and start the server
async function startServer() {
  try {
    await initializeVectorDB();
    await initializeEmbedder();
    
    const port = process.env.PORT || 3000;
    app.listen(port, () => {
      console.log(`Server is running on port ${port}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer(); 