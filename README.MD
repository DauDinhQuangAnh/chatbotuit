# Drag and Drop RAG

## Overview

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Usage Instructions

### Step 1: Upload Data
Upload a CSV. You can specify which columns to index for vector-based search.

### Step 2: Embedding and Storage
The data is stored in **Chroma**, and embeddings are generated using models like `keepitreal/vietnamese-sbert` (for Vietnamese).

### Step 3: LLM
- **Gemini API** (requires a [Gemini API key](https://aistudio.google.com/app/apikey))

### Step 4: Configure Chunking
Select a chunking method to organize the content:
- **Semantic Chunking**: Group text semantically.

### Step 5: Interact with the Chatbot
Start chatting with the bot, which will enhance responses using the retrieved content.

### Search Methods
- **Vector Search**: Based on vector similarity.
- **Hyde Search**: Uses a generative model for improved search accuracy.
