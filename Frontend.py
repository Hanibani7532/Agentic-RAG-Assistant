import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import SystemMessage

# Page Config (Website title and icon)
st.set_page_config(page_title="Hani AI Assistant", page_icon="⚡", layout="wide")

# Load Environment Variables
load_dotenv()

# CSS for better UI styling
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; padding: 10px; }
    .stChatInput { border-radius: 20px; }
    .stSidebar { padding: 1rem 1rem; }
    .stTitle { color: #008080; }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("⚡ Hani AI: Advanced Agentic RAG Assistant")
st.markdown("This AI agent can answer queries using both live internet search and your uploaded PDF document.")

# Initialize essential state components
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tools" not in st.session_state:
    st.session_state.tools = [TavilySearch(max_results=3)] # Default internet tool
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# System Prompt to instruct Llama to handle non-PDF questions gracefully
system_prompt = SystemMessage(content="""
    You are a helpful, professional AI Assistant. You have access to two tools: internet search (via TavilySearch) and a document search tool (if a document is uploaded).
    Your reasoning process is crucial:
    1. Analyze the user's query carefully.
    2. If the query is related to personal details, skills, experience, projects, or any content likely found within a professional document or CV, use the 'search_document' tool.
    3. If the query is about general knowledge, current events, weather, or anything unrelated to the document content, use the 'TavilySearch' tool.
    4. Do not ask the user to upload a document if they have not done so and the query does not require it. If a query requires the document and none is uploaded, gently inform them.
    5. Synthesized answers should be concise and polite.
""")

# Sidebar: Configuration and Dynamic Upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PDF Uploader
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file (e.g., CV or Report)", type=["pdf"])
    
    # LLM Model Selector (Optional - in case they want to switch)
    model_name = st.selectbox("Select LLM Model", ["llama3.1"], index=0)

    # Re-initialization logic when a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
        st.session_state.current_file = uploaded_file.name
        st.session_state.agent_ready = False # Mark agent as not ready while processing
        
        with st.status("📄 Reading PDF and building database, please wait...", expanded=True) as status:
            # Create a temporary path to process the PDF bytes
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 1. RAG Setup (PDF Load, Chunk, Store)
                status.write("A) Loading and splitting PDF...")
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)
                
                status.write("B) Creating vector database...")
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever()
                
                # 2. RAG Tool Setup
                pdf_tool = create_retriever_tool(
                    retriever,
                    "search_document",
                    f"Searches and returns information from the uploaded document: '{uploaded_file.name}'. Use this tool for queries about content within this document."
                )
                
                # Update Tools list with the new PDF tool
                st.session_state.tools = [TavilySearch(max_results=3), pdf_tool]
                status.write("C) Agent ready with new document tool!")
                status.update(label="✅ Document Database Ready!", state="complete", expanded=False)
                
                # Cleanup: Delete the temporary file
                os.remove(temp_path)
                st.session_state.agent_ready = True
                st.rerun() # Refresh the app to update the chat interface

            except Exception as e:
                st.error(f"❌ Error processing the PDF: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    elif uploaded_file is not None:
         # File already processed and current
         st.success(f"File loaded: **{st.session_state.current_file}**")
         st.session_state.agent_ready = True
    
    elif uploaded_file is None and st.session_state.current_file is not None:
         # User removed the file from uploader
         st.session_state.current_file = None
         st.session_state.tools = [TavilySearch(max_results=3)] # Default back to just internet
         st.session_state.agent_ready = False # Will rely purely on system message for defaults
         st.rerun()

# --- Main Chat Area ---

# LLM Setup (Needs to be inside main loop or session state to handle config changes)
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)

# Initialize/Get Agent Executor
try:
    if st.session_state.tools:
         agent_executor = create_react_agent(llm, st.session_state.tools)
    else:
         # Fallback just in case
         agent_executor = create_react_agent(llm, [TavilySearch(max_results=3)])
except Exception as e:
    st.error(f"❌ Agent setup error: {e}")
    agent_executor = None


# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and show a loading spinner
    if agent_executor:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is reasoning and searching..."):
                try:
                    # Get response from the agent (Passing system prompt here)
                    response = agent_executor.invoke({
                        "messages": [
                            system_prompt,
                            ("user", prompt)
                        ]
                    })
                    final_answer = response["messages"][-1].content
                    
                    # Display response and save to history
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")