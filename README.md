# ⚡ Hani AI: Advanced Agentic RAG Assistant

An intelligent, local AI Agent that seamlessly combines real-time internet search with local document retrieval (RAG). Built with **LangGraph** and powered by **Ollama (Llama 3.1)**, this agent can autonomously decide whether to search the web or read your uploaded PDF to answer queries accurately.

## 🌟 Key Features
* **Agentic Routing:** Automatically decides between using Web Search or Document Search based on the user's prompt.
* **Local & Private:** Uses local LLMs (Ollama) for complete privacy. No data is sent to OpenAI or Anthropic.
* **Dynamic Multi-Doc RAG:** Upload any PDF via the UI, and the system dynamically creates a local FAISS vector database on the fly.
* **Modern UI:** A clean, ChatGPT-like interface built with Streamlit, featuring a custom dark theme.

## 🛠️ Tech Stack
* **Framework:** LangGraph, LangChain
* **LLM:** Llama 3.1 (via Ollama)
* **Embeddings:** nomic-embed-text (via Ollama)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Web Search Tool:** Tavily API
* **Frontend:** Streamlit

## 🚀 Installation & Setup

### Prerequisites
1. Install [Python 3.10+](https://www.python.org/downloads/)
2. Install [Ollama](https://ollama.com/)
3. Get a free API key from [Tavily](https://tavily.com/)

### Step 1: Clone the Repository
```bash
git clone (https://github.com/Hanibani7532/Agentic-RAG-Assistant.git)
cd agentic-rag-assistant
Step 2: Create a Virtual Environment
Bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
Step 3: Install Dependencies
Bash
pip install -r requirements.txt
Step 4: Pull Local Models (Ollama)
Make sure Ollama is running, then download the required models in your terminal:

Bash
ollama pull llama3.1
ollama pull nomic-embed-text
Step 5: Environment Variables
Create a .env file in the root directory and add your Tavily API key:

Code snippet
TAVILY_API_KEY=tvly-your_api_key_here
OLLAMA_MODEL=llama3.1
Step 6: Run the Application
Bash
streamlit run Frontend.py
🧠 How it Works
The user inputs a query in the Streamlit UI.

The LangGraph ReAct Agent receives the prompt and the System Instructions.

If the query requires factual, real-time data, the agent invokes the TavilySearch tool.

If the query requires information from the uploaded document, the agent invokes the search_document tool (FAISS Retriever).

The LLM synthesizes the retrieved information and streams the final answer back to the UI.
