# 🤖 Multi-Modal RAG QA System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced **Multi-Modal Retrieval-Augmented Generation (RAG)** system designed to handle complex documents containing text, tables, images, charts, and figures. Built for real-world document intelligence challenges like financial reports, research papers, and technical manuals.

## 🎯 Features

### 📥 Multi-Modal Document Ingestion
- **PDF Processing**: Extract text, tables, charts, and images from PDFs
- **OCR Engine**: Handle scanned documents using PaddleOCR, Tesseract, and EasyOCR
- **Table Extraction**: Parse complex table structures with Camelot, Tabula, and PDFPlumber
- **Image Processing**: Analyze charts, diagrams, and figures
- **DOCX Support**: Process Word documents with full formatting

### 🧠 Intelligent Processing
- **Unified Embeddings**: Multi-modal vector space combining text and visual content
- **Semantic Chunking**: Context-aware document segmentation for optimal retrieval
- **Smart Retrieval**: Hybrid search combining dense and sparse methods
- **Cross-Modal Matching**: Find relevant information across different modalities

### 💬 Interactive QA Interface
- **Context-Grounded Answers**: Responses strictly based on uploaded documents
- **Source Attribution**: Page and section-level citations for transparency
- **Multi-Turn Conversations**: Maintain context across multiple queries
- **Real-Time Processing**: Interactive document upload and processing

## 🏗️ Architecture

```
┌─────────────────┐
│ Document Upload │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│   Multi-Modal Ingestion Layer    │
│  ├─ PDF Parser (PyMuPDF)         │
│  ├─ Table Extractor (Camelot)    │
│  ├─ OCR Engine (PaddleOCR)       │
│  └─ Image Processor (PIL)        │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│    Chunking & Embedding Layer     │
│  ├─ Semantic Text Chunking        │
│  ├─ Table Structure Preservation  │
│  ├─ Image Caption Generation      │
│  └─ CLIP Multi-Modal Embeddings   │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│       Vector Store (ChromaDB)     │
│  ├─ Text Embeddings              │
│  ├─ Table Embeddings             │
│  └─ Image Embeddings             │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│     Retrieval & Generation        │
│  ├─ Query Embedding              │
│  ├─ Hybrid Retrieval (Top-K)    │
│  ├─ Context Assembly             │
│  └─ LLM Generation (GPT-4)       │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Response with Citations          │
└──────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key (for LLM)
- Tesseract OCR (optional, for advanced OCR)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nvishnu-vardhan/multimodal-rag-qa-system.git
cd multimodal-rag-qa-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Add your OpenAI API key to .env file
OPENAI_API_KEY=your_api_key_here
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open your browser**
```
http://localhost:8501
```

## 📖 Usage

### 1. Upload Documents
- Click on the sidebar and upload your documents (PDF, DOCX, Images)
- Supports multiple file uploads

### 2. Configure Settings
- Enter your OpenAI API key
- Select LLM model (GPT-4, GPT-3.5-turbo, Gemini)
- Adjust chunk size and Top-K results

### 3. Process Documents
- Click "Process Documents" to start ingestion
- System will extract text, tables, and images
- Creates embeddings and stores in vector database

### 4. Ask Questions
- Type your question in the chat interface
- Receive context-grounded answers with citations
- View sources and page numbers for each answer

## 💻 Project Structure

```
multimodal-rag-qa-system/
├── app.py                      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
│
├── src/
│   ├── ingestion/              # Document ingestion modules
│   │   ├── __init__.py
│   │   ├── document_parser.py  # Main document parser
│   │   ├── pdf_processor.py    # PDF processing
│   │   ├── table_extractor.py  # Table extraction
│   │   ├── image_processor.py  # Image processing
│   │   └── ocr_engine.py       # OCR functionality
│   │
│   ├── chunking/               # Chunking strategies
│   │   ├── __init__.py
│   │   ├── text_chunker.py
│   │   ├── table_chunker.py
│   │   └── semantic_chunker.py
│   │
│   ├── embeddings/             # Embedding generation
│   │   ├── __init__.py
│   │   ├── text_embeddings.py
│   │   ├── multimodal_embeddings.py
│   │   └── clip_embeddings.py
│   │
│   ├── retrieval/              # Retrieval system
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── generation/             # Answer generation
│   │   ├── __init__.py
│   │   ├── qa_chain.py
│   │   ├── prompt_templates.py
│   │   └── citation_builder.py
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── file_handler.py
│       └── logger.py
│
├── data/                       # Data storage
│   ├── uploads/               # Uploaded documents
│   ├── processed/             # Processed data
│   └── vector_db/             # Vector database storage
│
├── tests/                      # Unit tests
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── docs/                       # Additional documentation
    ├── ARCHITECTURE.md
    ├── API_REFERENCE.md
    └── DEPLOYMENT.md
```

## 🔧 Technology Stack

### Core Framework
- **Streamlit**: Interactive web interface
- **LangChain**: RAG pipeline orchestration
- **Python 3.8+**: Backend language

### LLM & Embeddings
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Sentence Transformers**: Text embeddings
- **CLIP**: Multi-modal embeddings
- **Google Gemini**: Alternative LLM support

### Document Processing
- **PyMuPDF (fitz)**: PDF parsing
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **Pillow**: Image processing

### Table Extraction
- **Camelot**: Advanced table extraction
- **Tabula**: PDF table parsing
- **PDFPlumber**: Layout-aware extraction
- **Pandas**: Data manipulation

### OCR Engines
- **PaddleOCR**: High-accuracy OCR
- **Tesseract**: Open-source OCR
- **EasyOCR**: Multi-language support

### Vector Database
- **ChromaDB**: Primary vector store
- **FAISS**: Fast similarity search
- **Pinecone**: Cloud vector database (optional)

## 📊 Use Cases

### 🏦 Financial Analysis
Query annual reports, balance sheets, and financial statements with complex tables and charts.

### 🔬 Research Papers
Extract insights from academic papers with equations, figures, and citations.

### ⚖️ Legal Documents
Search through contracts, policies, and legal briefs with dense text and tables.

### 📘 Technical Manuals
Find specific procedures, diagrams, and technical specifications.

### 📈 Business Intelligence
Analyze presentations, reports, and dashboards with mixed content types.

## 🎓 Assignment Compliance

This project fulfills all requirements of the Multi-Modal Document Intelligence RAG-Based QA System assignment:

### ✅ Features Implemented
- [x] Multi-modal ingestion (text, tables, images, OCR)
- [x] Unified multi-modal embedding space
- [x] Semantic and structural chunking
- [x] Interactive QA chatbot interface
- [x] Page and section-level citations
- [x] Context-grounded answer generation

### ✅ Deliverables
- [x] Well-structured, modular codebase
- [x] Streamlit demo application
- [x] Comprehensive documentation
- [x] Clear setup instructions

### 🏆 Bonus Features
- Cross-modal retrieval
- Hybrid search with reranking
- Multiple LLM support (GPT-4, GPT-3.5, Gemini)
- Interactive statistics dashboard
- Document processing status tracking

## 🚧 Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Vercel Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

### Docker Deployment
```bash
# Build image
docker build -t multimodal-rag .

# Run container
docker run -p 8501:8501 multimodal-rag
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vishnu Vardhan**
- GitHub: [@nvishnu-vardhan](https://github.com/nvishnu-vardhan)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/nvishnu-vardhan)

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for RAG framework
- Streamlit for the amazing web framework
- The open-source community for various libraries

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API](https://platform.openai.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**Built with ❤️ for advanced document intelligence**

⭐ Star this repository if you find it helpful!
