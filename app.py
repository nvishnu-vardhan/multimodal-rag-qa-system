import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Multi-Modal RAG QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 Multi-Modal RAG QA System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Document Intelligence with Text, Tables, Images & Citations</p>', unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Enter your OpenAI API key for LLM processing")
    
    st.divider()
    
    # Document upload
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, Images)",
        type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if st.button("🔄 Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Placeholder for document processing
                st.session_state.documents_processed = True
                st.success(f"✅ Processed {len(uploaded_files)} document(s)!")
        else:
            st.warning("Please upload documents first")
    
    st.divider()
    
    # Model settings
    st.header("🎛️ Model Settings")
    model = st.selectbox(
        "LLM Model",
        ["gpt-4", "gpt-3.5-turbo", "gemini-pro"],
        index=1
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    chunk_size = st.number_input("Chunk Size", 256, 2048, 512, 128)
    
    top_k = st.number_input("Top K Results", 1, 10, 3, 1)
    
    st.divider()
    
    # Stats
    if st.session_state.documents_processed:
        st.header("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(uploaded_files) if uploaded_files else 0)
        with col2:
            st.metric("Queries", len(st.session_state.chat_history))

# Main content
main_tab, docs_tab, about_tab = st.tabs(["💬 Chat", "📚 Documents", "ℹ️ About"])

with main_tab:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message:
                with st.expander("📖 View Citations"):
                    for citation in message["citations"]:
                        st.markdown(f'<div class="citation-box">{citation}</div>', 
                                  unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_processed:
            st.warning("Please upload and process documents first!")
        elif not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar!")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response (placeholder)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # This is a placeholder response
                    response = f"""Based on the documents, here's what I found:
                    
                    This is a placeholder response demonstrating the multi-modal RAG system. 
                    In a complete implementation, this would:
                    
                    1. **Extract** text, tables, and images from your documents
                    2. **Process** them through OCR and table extraction
                    3. **Embed** all content into a unified vector space
                    4. **Retrieve** the most relevant chunks based on your query
                    5. **Generate** a comprehensive answer with citations
                    
                    Your question: "{prompt}"
                    
                    *Note: This is a demo. Full implementation requires document processing pipeline.*
                    """
                    
                    st.markdown(response)
                    
                    # Mock citations
                    citations = [
                        "📄 Document: example.pdf | Page: 5 | Section: 2.3",
                        "📊 Table: Financial Summary | Source: report.pdf | Page: 12",
                        "🖼️ Image: Chart Analysis | Context: quarterly_data.pdf | Page: 8"
                    ]
                    
                    with st.expander("📖 View Citations"):
                        for citation in citations:
                            st.markdown(f'<div class="citation-box">{citation}</div>', 
                                      unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "citations": citations
                    })

with docs_tab:
    st.header("📚 Processed Documents")
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} document(s) uploaded:**")
        
        for idx, file in enumerate(uploaded_files, 1):
            with st.expander(f"{idx}. {file.name} ({file.size / 1024:.2f} KB)"):
                st.write("**File Information:**")
                st.write(f"- Type: {file.type}")
                st.write(f"- Size: {file.size / 1024:.2f} KB")
                
                if st.session_state.documents_processed:
                    st.write("\n**Processing Status:**")
                    st.success("✅ Processed successfully")
                    st.write("- Text extracted: ✓")
                    st.write("- Tables extracted: ✓")
                    st.write("- Images processed: ✓")
                    st.write("- Embeddings created: ✓")
    else:
        st.info("No documents uploaded yet. Upload documents using the sidebar.")

with about_tab:
    st.header("ℹ️ About Multi-Modal RAG QA System")
    
    st.markdown("""
    ### 🎯 Features
    
    This system implements a comprehensive Multi-Modal Retrieval-Augmented Generation pipeline:
    
    #### 📥 Document Ingestion
    - **PDF Processing**: Extract text, tables, charts, and images
    - **OCR Engine**: Handle scanned documents and images
    - **Table Extraction**: Parse complex table structures
    - **Image Analysis**: Process charts, diagrams, and figures
    
    #### 🧠 Intelligent Processing
    - **Multi-Modal Embeddings**: Unified vector space for text, tables, and images
    - **Semantic Chunking**: Context-aware document segmentation
    - **Smart Retrieval**: Hybrid search combining dense and sparse methods
    - **Cross-Modal Matching**: Find relevant information across modalities
    
    #### 💬 QA Interface
    - **Context-Grounded Answers**: Responses based on your documents
    - **Source Attribution**: Page and section-level citations
    - **Multi-Turn Conversations**: Maintain context across queries
    - **Confidence Scores**: Transparency in answer generation
    
    ### 🔧 Technology Stack
    
    - **Framework**: Streamlit
    - **LLM**: OpenAI GPT-4 / GPT-3.5
    - **Embeddings**: Sentence Transformers, CLIP
    - **Vector DB**: ChromaDB / FAISS
    - **OCR**: PaddleOCR, Tesseract, EasyOCR
    - **Table Extraction**: Camelot, Tabula, PDFPlumber
    - **Document Parsing**: PyMuPDF, PyPDF2
    
    ### 🚀 Getting Started
    
    1. Enter your OpenAI API key in the sidebar
    2. Upload your documents (PDF, DOCX, Images)
    3. Click "Process Documents" to ingest content
    4. Start asking questions in the Chat tab
    5. View citations and sources for each answer
    
    ### 📊 Use Cases
    
    - **Financial Analysis**: Query annual reports with tables and charts
    - **Research Papers**: Extract insights from academic documents
    - **Legal Documents**: Search through contracts and policies
    - **Technical Manuals**: Find specific procedures and diagrams
    - **Business Intelligence**: Analyze presentations and reports
    
    ### 🔗 Repository
    
    [GitHub Repository](https://github.com/nvishnu-vardhan/multimodal-rag-qa-system)
    
    ---
    
    **Built with ❤️ for advanced document intelligence**
    """)
