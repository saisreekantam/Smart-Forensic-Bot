# Smart Forensic Bot 🔍

An AI-powered digital forensics analysis system with LangGraph, RAG (Retrieval-Augmented Generation), and knowledge graph integration.

## Features 🚀

- **🤖 AI-Powered Analysis**: Leverages OpenAI's GPT models for intelligent forensic analysis
- **📊 Knowledge Graph**: Builds and maintains relationships between forensic evidence
- **🔍 RAG Integration**: Retrieval-Augmented Generation for contextual analysis
- **⚡ LangGraph Workflows**: Automated forensic investigation workflows
- **📁 File Analysis**: Comprehensive file analysis with hash verification and metadata extraction
- **🌐 Network Forensics**: Analysis of network activities and relationships
- **💾 Vector Database**: Semantic search capabilities for forensic knowledge
- **📋 Rich CLI**: Beautiful command-line interface with detailed reporting
- **📈 Timeline Analysis**: Event correlation and timeline reconstruction

## Architecture 🏗️

```
Smart Forensic Bot
├── Core Configuration
├── Forensics Analysis Engine
│   ├── File Analyzer
│   ├── Hash Calculator
│   └── Metadata Extractor
├── RAG System
│   ├── Vector Store (FAISS)
│   ├── Embedding Model
│   └── Knowledge Retrieval
├── Knowledge Graph
│   ├── Entity Management
│   ├── Relationship Mapping
│   └── Timeline Construction
├── LangGraph Workflows
│   ├── Analysis Pipeline
│   ├── Evidence Correlation
│   └── Report Generation
└── CLI Interface
    ├── Analysis Commands
    ├── Search Functions
    └── Reporting Tools
```

## Installation 📦

1. **Clone the repository**:
```bash
git clone https://github.com/saisreekantam/Smart-Forensic-Bot.git
cd Smart-Forensic-Bot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Initialize the system**:
```bash
python main.py init
```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key and other configurations

## Quick Start 🏃‍♂️

### Initialize the System
```bash
python main.py init
```

### Import Knowledge Base
```bash
python main.py import-knowledge examples/knowledge_base.json
```

### Analyze Evidence
```bash
# Analyze a single file
python main.py scan /path/to/evidence/file.exe

# Analyze a directory
python main.py analyze /path/to/evidence/ --case-id CASE001 --analyst "John Doe"

# Analyze with output to file
python main.py analyze /path/to/evidence/ --case-id CASE001 --output report.json
```

### Search Knowledge Base
```bash
python main.py search "malware analysis"
```

### View Knowledge Graph Statistics
```bash
python main.py graph-stats
```

## Usage Examples 💡

### Complete Analysis Workflow
```bash
# Run the demo analysis
./examples/demo_analysis.sh
```

### Advanced Analysis
```bash
# Analyze with specific parameters
python main.py analyze evidence/ \
  --case-id "INCIDENT_2024_001" \
  --analyst "Senior Analyst" \
  --output "reports/detailed_analysis.json" \
  --recursive
```

### Knowledge Base Management
```bash
# Import forensic knowledge
python main.py import-knowledge knowledge/forensic_patterns.json

# Search for specific patterns
python main.py search "registry persistence" --limit 10
```

## Configuration ⚙️

### Environment Variables
Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Paths
VECTORDB_PATH=./data/vectordb
EVIDENCE_PATH=./evidence
TEMP_PATH=./temp
REPORTS_PATH=./reports
LANGGRAPH_CHECKPOINT_PATH=./checkpoints
```

### Knowledge Base Format
```json
[
  {
    "content": "Description of forensic technique or knowledge",
    "category": "file_analysis",
    "source": "forensic_handbook"
  }
]
```

## API Reference 📚

### Core Components

#### FileAnalyzer
```python
from src.smart_forensic_bot.forensics.file_analyzer import FileAnalyzer

analyzer = FileAnalyzer()
analysis = analyzer.analyze_file(Path("evidence.exe"))
```

#### ForensicVectorStore
```python
from src.smart_forensic_bot.rag.vector_store import ForensicVectorStore

vector_store = ForensicVectorStore()
results = vector_store.search("malware indicators", k=5)
```

#### ForensicKnowledgeGraph
```python
from src.smart_forensic_bot.knowledge_graph.graph import ForensicKnowledgeGraph

graph = ForensicKnowledgeGraph()
graph.add_forensic_evidence(evidence_data)
related = graph.find_related_entities("evidence_id")
```

#### ForensicWorkflow
```python
from src.smart_forensic_bot.workflows.forensic_analysis import ForensicWorkflow

workflow = ForensicWorkflow()
results = workflow.run_analysis(
    evidence_files=["file1.exe", "file2.dll"],
    case_id="CASE001",
    analyst="Analyst Name"
)
```

## Supported File Types 📄

- **Executables**: PE files (.exe, .dll)
- **Documents**: PDF, Word, Excel files
- **Images**: JPEG, PNG, GIF
- **Archives**: ZIP, RAR, TAR
- **Logs**: Text files, CSV files
- **Scripts**: Batch files, PowerShell scripts
- **And many more...**

## Analysis Features 🔬

### File Analysis
- **Hash Calculation**: MD5, SHA1, SHA256
- **File Type Detection**: MIME type and description
- **Metadata Extraction**: Timestamps, permissions, ownership
- **Suspicious Indicators**: Detection of potentially malicious patterns

### Knowledge Integration
- **Vector Search**: Semantic search through forensic knowledge
- **Graph Relationships**: Entity and relationship mapping
- **Timeline Construction**: Chronological event analysis
- **Context Retrieval**: Relevant knowledge for analysis

### AI-Powered Insights
- **Intelligent Analysis**: GPT-powered forensic insights
- **Pattern Recognition**: AI-driven pattern detection
- **Report Generation**: Automated report creation
- **Recommendations**: Actionable investigation steps

## Example Output 📊

```
🔍 Smart Forensic Bot Analysis Results

Case Information:
┌─────────────┬──────────────────────────┐
│ Field       │ Value                    │
├─────────────┼──────────────────────────┤
│ Case ID     │ CASE_20241201_143022     │
│ Analyst     │ Demo Analyst             │
│ Timestamp   │ 2024-12-01T14:30:22      │
│ Files       │ 3                        │
└─────────────┴──────────────────────────┘

🔍 Key Findings:
1. Suspicious executable detected with unusual entry point
2. Registry persistence mechanism identified
3. Network communication to known malicious IP
4. Timestamps indicate coordinated attack timeline

💡 Recommendations:
1. Isolate affected systems immediately
2. Block identified malicious IP addresses
3. Scan for additional infected systems
4. Preserve evidence for legal proceedings
```

## Contributing 🤝

We welcome contributions! Please see our contributing guidelines for more information.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/saisreekantam/Smart-Forensic-Bot.git
cd Smart-Forensic-Bot
pip install -r requirements.txt
python main.py init
```

### Running Tests
```bash
# Run the test suite (when available)
pytest tests/
```

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer ⚠️

This tool is for educational and authorized forensic analysis purposes only. Always ensure you have proper authorization before analyzing any systems or data.

## Support 🆘

- 📧 Email: support@smartforensicbot.com
- 🐛 Issues: [GitHub Issues](https://github.com/saisreekantam/Smart-Forensic-Bot/issues)
- 📖 Documentation: [Wiki](https://github.com/saisreekantam/Smart-Forensic-Bot/wiki)

## Roadmap 🗺️

- [ ] Web-based dashboard interface
- [ ] Integration with YARA rules
- [ ] Memory forensics capabilities
- [ ] Mobile device analysis
- [ ] Cloud forensics support
- [ ] Machine learning model training
- [ ] Integration with SIEM systems
- [ ] Automated threat hunting

---

**Smart Forensic Bot** - Revolutionizing digital forensics with AI 🚀