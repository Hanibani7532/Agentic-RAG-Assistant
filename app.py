import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

# 1. Load environment variables
load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    print("❌ ERROR: TAVILY_API_KEY not found!")
    sys.exit()

# 2. RAG Setup: Read PDF and create database
print("📄 Reading CV, please wait...")

# Path to your file
pdf_path = "data/Hani AI ML Engineer.pdf"

try:
    # A) Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # B) Split text into smaller chunks (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # C) Create embeddings and FAISS Vector Store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    # D) Create RAG Tool (To be used by the Agent)
    pdf_tool = create_retriever_tool(
        retriever,
        "search_hani_cv",
        "Searches and returns information from Hani's AI ML Engineer CV/Resume. Use this tool when asked about Hani's skills, experience, or projects."
    )
    print("✅ CV Database is ready!\n")
except Exception as e:
    print(f"❌ Error reading PDF: {e}")
    sys.exit()


# 3. LLM Setup (Ollama Llama 3.1)
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)

# 4. Tools Setup (We now have 2 tools!)
internet_tool = TavilySearch(max_results=3)
tools = [internet_tool, pdf_tool]

# 5. Create Agent
agent_executor = create_react_agent(llm, tools)

if __name__ == "__main__":
    print("--- ⚡ Agentic RAG (Internet + CV Data) is Ready! ---")
    while True:
        user_input = input("\nWhat would you like to ask? (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
            
        print("\n🤖 Agent is thinking...\n")
        
        # Run the agent
        response = agent_executor.invoke({"messages": [("user", user_input)]})
        
        # Print the last message (Final Answer)
        print("🎯 Final Result:\n", response["messages"][-1].content)