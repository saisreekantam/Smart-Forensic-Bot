# Smart Forensic Investigation Platform

A comprehensive AI-powered forensic investigation platform that enables law enforcement professionals to manage cases, process digital evidence, and conduct intelligent analysis through natural language interactions. The platform combines case management, evidence processing, and advanced LLM-powered chatbot capabilities.

# Note:

The Current prototype is limited to XML data without GPS values and messages. However, the final product will support these additional data fields.

##  Visual Resources

###  Application Screenshots
**Google Drive Gallery**: [View Webapp Screenshots & UI Demo](https://drive.google.com/drive/folders/1ANICO674D5fWbad-WiDfQxbLe4tObIpw?usp=sharing)

### Video Demonstration
**Link of Google drive with Video**: [View Webapp Video Demonstration](https://drive.google.com/drive/folders/1NqEZTS1O67dbu3scGDogHwdtlbXQqZW9?usp=sharing)

### Presentation Slides
**PPT Google Drive link**: [View Presentation](https://drive.google.com/drive/folders/1Nl7g1yU6QBPpvbr3oDtv_Gm7UF9xtAt-?usp=sharing)

---

##  Project Overview

**Project Sentinel** is a complete case-based forensic investigation platform that transforms traditional digital forensics workflows. The system provides:

- **Case Management**: Create, organize, and track forensic investigations
- **Evidence Processing**: Upload and analyze multiple evidence formats (XML, CSV, JSON, PDF, TXT)
- **AI-Powered Analysis**: Natural language chatbot for querying evidence using advanced LLMs
- **Chat History**: Persistent conversation sessions with case context
- **Smart Search**: Intelligent evidence search with fuzzy matching and relevance scoring
- **Real-time Processing**: Background evidence processing with status tracking

## System Architecture

### Backend Stack
- **FastAPI**: RESTful API framework with automatic documentation
- **SQLAlchemy**: Database ORM for case and evidence management
- **SQLite**: Primary database for development (PostgreSQL ready)
- **ChromaDB**: Vector database for semantic search
- **LangChain**: LLM framework and RAG implementation
- **Sentence Transformers**: Text embeddings for semantic search

### Frontend Stack
- **React 18**: Modern frontend framework with TypeScript
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/UI**: Component library for consistent design
- **React Query**: Server state management
- **React Router**: Client-side routing

### AI & ML Technologies
- **LLM Integration**: Primary (Gemini 2.5 Pro), Secondary (GPT-4), Fallback (Rule-based)
- **Vector Embeddings**: Sentence transformers for semantic search
- **RAG System**: Retrieval-Augmented Generation for evidence-based responses
- **Fuzzy Search**: Advanced text matching with relevance scoring
- **Entity Recognition**: Phone numbers, emails, cryptocurrency addresses

### Data Processing
- **Multi-format Support**: XML, CSV, JSON, PDF, TXT evidence files
- **Intelligent Chunking**: Smart text segmentation for optimal embedding
- **Entity Extraction**: Automated identification of forensic entities
- **PDF Processing**: Advanced extraction with PyPDF2
- **Background Processing**: Asynchronous evidence processing

##  Project Structure

```
Smart-Forensic-Investigation-Platform/
├── src/
│   ├── api/
│   │   └── case_api.py                    # FastAPI endpoints & chat management
│   ├── ai_cores/
│   │   ├── enhanced_assistant.py          # Smart conversation classification
│   │   ├── langgraph_assistant.py         # Advanced AI processing
│   │   ├── case_memory.py                 # Case context management
│   │   ├── enhanced_knowledge_graph.py    # Knowledge graph integration
│   │   └── intelligent_report_generator.py # Automated reporting
│   ├── case_management/
│   │   └── case_manager.py                # Core case operations
│   ├── data_ingestion/
│   │   ├── case_processor.py              # Evidence processing pipeline
│   │   ├── evidence_handlers.py           # File type handlers
│   │   ├── parsers.py                     # UFDR data parsers
│   │   ├── preprocessor.py                # Text preprocessing
│   │   └── chunking.py                    # Intelligent text chunking
│   ├── database/
│   │   └── models.py                      # SQLAlchemy database models
│   └── frontend/
│       ├── src/
│       │   ├── components/
│       │   │   ├── dashboard/             # Dashboard components
│       │   │   ├── layout/                # Layout components
│       │   │   └── ui/                    # Reusable UI primitives
│       │   ├── hooks/                     # Custom React hooks
│       │   ├── lib/                       # Utility functions
│       │   └── pages/
│       │       ├── Chatbot.tsx            # AI chatbot interface
│       │       ├── Dashboard.tsx          # Main dashboard
│       │       ├── Upload.tsx             # Evidence upload
│       │       ├── Reports.tsx            # Investigation reports
│       │       └── Analytics.tsx          # Case analytics
│       ├── package.json                   # Frontend dependencies
│       ├── vite.config.ts                 # Vite configuration
│       └── tailwind.config.ts             # Tailwind CSS config
├── data/
│   ├── cases/                             # Case-specific evidence storage
│   ├── vector_db/                         # ChromaDB embeddings
│   └── processed/                         # Processing results
├── simple_chat_handler.py                 # Core chat processing logic
├── simple_search_system.py                # Enhanced evidence search
├── simple_data_processor.py               # Multi-format data processing
├── chat_history_manager.py                # Chat session persistence
├── start_api.py                           # FastAPI server startup
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

##  Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/saisreekantam/Smart-Forensic-Bot.git
cd Smart-Forensic-Bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model for NLP
python -m spacy download en_core_web_sm

# Set up environment variables (optional but recommended)
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd src/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 3. Start the Application

```bash
# Start backend API server
python case_api.py
# Backend runs on http://localhost:8000

# Frontend runs on http://localhost:3000 (if started separately)
```

### 4. Access the Platform

- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Frontend Application**: http://localhost:3000

##  Key Features

### Case Management
- Create and organize forensic investigations
- Track case status, priority, and metadata
- Multi-investigator support with access control
- Case statistics and progress tracking

### Evidence Processing
- **Supported Formats**: XML, CSV, JSON, PDF, TXT, UFDR files
- **Automatic Processing**: Background file processing with status updates
- **Entity Extraction**: Phone numbers, emails, crypto addresses, names
- **Intelligent Chunking**: Optimized text segmentation for AI analysis

### AI-Powered Chatbot
- **Natural Language Queries**: Ask questions about evidence in plain English
- **Context-Aware Responses**: Maintains conversation context and case awareness
- **Multi-LLM Support**: Primary (Gemini), backup (GPT-4), fallback (rule-based)
- **Source Attribution**: All responses cite specific evidence sources
- **Chat Sessions**: Persistent conversation history with case context

### Advanced Search
- **Semantic Search**: Vector-based similarity search across all evidence
- **Fuzzy Matching**: Find relevant content even with typos or variations
- **Multi-format Search**: Search across all evidence types simultaneously
- **Relevance Scoring**: Advanced algorithm for ranking search results

### Frontend Interface
- **Modern UI**: Clean, professional interface built with React and Tailwind
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live status updates for processing and analysis
- **Interactive Dashboard**: Visual case overview with statistics and progress

##  Chatbot Capabilities

The AI assistant can help with:

### Evidence Analysis
```
"What evidence do we have in this case?"
"Show me all messages from John Smith"
"Find suspicious cryptocurrency transactions"
```

### Communication Analysis
```
"Who talked to Alex Rivera and when?"
"Show me threatening messages"
"Find all communications on January 15th"
```

### Pattern Detection
```
"Are there any patterns in the call logs?"
"Identify suspicious activities"
"Show connections between contacts"
```

### Investigation Guidance
```
"What should I investigate next?"
"Summarize key findings"
"Generate investigation timeline"
```

##  Database Schema

### Core Tables

**Cases**
- Case metadata, investigator info, status tracking
- Evidence counts and processing progress
- Vector store collection mapping

**Evidence**
- File information, processing status, embeddings metadata
- Chain of custody, source device information
- File integrity hashing

**Evidence Chunks**
- Processed text segments with embeddings
- Entity extraction results
- Conversation participants and timestamps

**Chat Sessions**
- Persistent conversation history
- Case-specific chat sessions
- Message threading and timestamps

##  API Endpoints

### Case Management
```http
POST /cases                    # Create new case
GET /cases                     # List all cases
GET /cases/{case_id}          # Get case details
PUT /cases/{case_id}          # Update case
DELETE /cases/{case_id}       # Delete case
```

### Evidence Management
```http
POST /cases/{case_id}/evidence          # Upload evidence
GET /cases/{case_id}/evidence           # List case evidence
POST /cases/{case_id}/process           # Process evidence
GET /cases/{case_id}/evidence/stats     # Evidence statistics
```

### Chat & AI Analysis
```http
POST /cases/{case_id}/chat                           # Send chat message
GET /cases/{case_id}/chat/sessions                   # List chat sessions
POST /cases/{case_id}/chat/sessions                  # Create chat session
GET /cases/{case_id}/chat/sessions/{session_id}/messages  # Get messages
DELETE /cases/{case_id}/chat/sessions/{session_id}   # Delete session
```

##  Configuration

### Environment Variables
```bash
# LLM API Keys (optional but recommended)
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=sqlite:///data/forensic_cases.db
# DATABASE_URL=postgresql://user:pass@host:port/dbname  # For production

# Vector Database
CHROMA_PERSIST_DIRECTORY=data/vector_db

# Application Settings
DEBUG_FORENSIC_BOT=false
MAX_UPLOAD_SIZE=100MB
PROCESSING_WORKERS=4
```

### Frontend Configuration
```typescript
// src/frontend/src/config.ts
export const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:8000';
```

##  Testing

### Backend Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Test specific modules
pytest tests/test_case_manager.py
pytest tests/test_search_system.py
pytest tests/test_chat_handler.py
```

### Frontend Tests
```bash
cd src/frontend

# Run unit tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

##  Troubleshooting

### Common Issues

#### Chat History Not Saving
**Symptoms**: "No chat sessions yet" message persists
**Solution**: 
- Check if backend server is running on port 8000
- Verify database file `data/forensic_cases.db` exists
- Check browser console for API errors

#### PDF Files Not Processing
**Symptoms**: PDF evidence not appearing in search results
**Solution**:
```bash
pip install PyPDF2
# Restart the server
python case_api.py
```

#### Search Results Too Broad/Narrow
**Symptoms**: Irrelevant results or missing evidence
**Solution**: Adjust relevance thresholds in `simple_search_system.py`:
```python
# Line ~120
if score >= 0.6:  # Lower for broader results, higher for stricter
    results.append(...)
```




## Acknowledgments

- **Forensic Community**: For domain expertise and requirements
- **AI/ML Libraries**: LangChain, OpenAI, Google AI, Sentence Transformers
- **Frontend Tools**: React, Tailwind CSS, Shadcn/UI
- **Backend Tools**: FastAPI, SQLAlchemy, ChromaDB

---

**Project Status**: Active Development
**Version**: 2.0.0
**Last Updated**: September 27, 2025


