# Knowledge Graph Generator

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced, flexible tool for automatically generating interactive knowledge graphs from documents using Large Language Models (LLMs). Supports multiple LLM providers including OpenAI, Anthropic, and Ollama.

## 🌟 Features

- **Multi-Provider LLM Support**: Choose from OpenAI (GPT-4, GPT-3.5), Anthropic (Claude), or local models via Ollama
- **Intelligent Concept Extraction**: Automatically identifies key concepts, entities, and relationships from documents
- **Interactive Visualizations**: Beautiful, interactive HTML graphs with community detection and customizable styling
- **Flexible Configuration**: YAML-based configuration with CLI overrides
- **Multiple Input Formats**: Supports PDF and TXT documents
- **Batch Processing**: Efficient processing of large documents with configurable batch sizes
- **Rich Output Formats**: Export to HTML, CSV, and JSON formats
- **Graph Analytics**: Built-in graph statistics and community detection

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Knowledge_graph.git
cd Knowledge_graph
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Ollama (if not using default)
OLLAMA_HOST=http://localhost:11434
```

### Step 4: Set Up Ollama (Optional)

If you want to use local models via Ollama:

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2
```

## ⚡ Quick Start

Generate a knowledge graph from a PDF document:

```bash
python knowledge_graph.py path/to/your/document.pdf
```

This will:
1. Extract concepts from your document
2. Build a knowledge graph
3. Create an interactive visualization in `output/knowledge_graph.html`

Open the HTML file in your browser to explore the interactive graph!

## ⚙️ Configuration

The main configuration file is `config.yaml`. Here's what you can customize:

### LLM Settings

```yaml
llm:
  provider: "ollama"  # Options: "openai", "anthropic", "ollama"
  models:
    openai: "gpt-4o-mini"
    anthropic: "claude-3-5-sonnet-20241022"
    ollama: "llama3.2"
  temperature: 0.1
  max_tokens: 2000
```

### Document Processing

```yaml
document_processing:
  chunk_size: 1000        # Size of text chunks
  chunk_overlap: 100      # Overlap between chunks
  batch_size: 5           # Documents per batch
  delay_between_batches: 2 # Seconds delay
```

### Graph Settings

```yaml
graph:
  min_importance: 2           # Min importance score (1-5)
  node_size_multiple: 6       # Node size multiplier
  enable_communities: true     # Detect communities
  visualization:
    height: "900px"
    width: "100%"
    background_color: "#1a1a1a"
    font_color: "#cccccc"
```

## 📖 Usage

### Basic Usage

```bash
# Using default configuration
python knowledge_graph.py document.pdf

# Specify output directory
python knowledge_graph.py document.pdf --output ./my_output

# Use a different config file
python knowledge_graph.py document.pdf --config custom_config.yaml
```

### Using Different LLM Providers

```bash
# Use OpenAI GPT-4
python knowledge_graph.py document.pdf --provider openai --model gpt-4

# Use Anthropic Claude
python knowledge_graph.py document.pdf --provider anthropic --model claude-3-5-sonnet-20241022

# Use Ollama with a specific model
python knowledge_graph.py document.pdf --provider ollama --model mistral
```

### Advanced Options

```bash
# Adjust chunk size for processing
python knowledge_graph.py document.pdf --chunk-size 500

# Change importance threshold
python knowledge_graph.py document.pdf --min-importance 3

# Disable community detection
python knowledge_graph.py document.pdf --no-communities

# Enable verbose logging
python knowledge_graph.py document.pdf --verbose
```

### Full CLI Options

```
usage: knowledge_graph.py [-h] [-o OUTPUT] [-c CONFIG]
                         [--provider {openai,anthropic,ollama}]
                         [--model MODEL] [--chunk-size CHUNK_SIZE]
                         [--min-importance MIN_IMPORTANCE]
                         [--no-communities] [--verbose]
                         input_file

positional arguments:
  input_file            Path to input document (PDF or TXT)

optional arguments:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output directory
  -c, --config CONFIG   Path to configuration file
  --provider {openai,anthropic,ollama}
                        LLM provider (overrides config)
  --model MODEL         Model name (overrides config)
  --chunk-size CHUNK_SIZE
                        Document chunk size (overrides config)
  --min-importance MIN_IMPORTANCE
                        Minimum importance threshold for graph (1-5)
  --no-communities      Disable community detection
  --verbose             Enable verbose logging (DEBUG level)
```

## 📊 Examples

### Example 1: Research Paper Analysis

```bash
python knowledge_graph.py research_paper.pdf --provider openai --model gpt-4
```

This will extract key concepts, methodologies, and findings from a research paper and visualize their relationships.

### Example 2: Book Chapter Analysis

```bash
python knowledge_graph.py chapter.txt --chunk-size 800 --min-importance 3
```

Processes a text file with smaller chunks and only includes highly important concepts.

### Example 3: Using Local Models

```bash
# Make sure Ollama is running
ollama serve

# Generate graph with local model
python knowledge_graph.py document.pdf --provider ollama --model llama3.2
```

### Example 4: Batch Processing Multiple Documents

```bash
# Process multiple documents
for file in documents/*.pdf; do
    python knowledge_graph.py "$file" --output "output/$(basename "$file" .pdf)"
done
```

## 📁 Output Files

After processing, you'll find these files in your output directory:

| File | Description |
|------|-------------|
| `knowledge_graph.html` | Interactive visualization (open in browser) |
| `graph.csv` | Graph edges in CSV format |
| `graph.json` | Graph data in JSON format |
| `graph_stats.json` | Graph statistics (nodes, edges, etc.) |
| `concepts.csv` | Extracted concepts with metadata |
| `chunks.csv` | Document chunks used for processing |

### Understanding the Visualization

The interactive HTML visualization includes:

- **Nodes**: Concepts/entities (size = importance)
- **Edges**: Relationships (thickness = co-occurrence frequency)
- **Colors**: Communities detected in the graph
- **Hover**: Shows detailed information
- **Physics Controls**: Adjust layout in real-time

## 🔧 Advanced Usage

### Custom Configuration

Create a custom config file for specific use cases:

```yaml
# scientific_papers.yaml
llm:
  provider: "openai"
  models:
    openai: "gpt-4"
  temperature: 0.0  # More deterministic

document_processing:
  chunk_size: 1500  # Larger chunks for technical content
  batch_size: 3

graph:
  min_importance: 3  # Only most important concepts
```

Use it:

```bash
python knowledge_graph.py paper.pdf --config scientific_papers.yaml
```

### Programmatic Usage

You can also use the library programmatically:

```python
from src.config import Config
from src.models import LLMFactory
from src.utils import DocumentProcessor
from src.extractors import ConceptExtractor, GraphBuilder

# Load config
config = Config('config.yaml')

# Initialize LLM
llm_client = LLMFactory.from_config(config)

# Process document
doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
chunks_df = doc_processor.process_file('document.pdf')

# Extract concepts
extractor = ConceptExtractor(llm_client)
concepts_df = extractor.extract_from_dataframe(chunks_df)

# Build graph
builder = GraphBuilder(min_importance=2)
graph_df = builder.build_graph_dataframe(concepts_df)

# Visualize
builder.visualize(graph_df, 'output/graph.html')
```

## 🐛 Troubleshooting

### Common Issues

**1. "API key not found" error**

Make sure you've created a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env and add your keys
```

**2. "Ollama service not available"**

Start the Ollama service:
```bash
ollama serve
```

**3. "No concepts extracted"**

Try adjusting these settings:
- Increase `temperature` in config.yaml (e.g., 0.3)
- Decrease `chunk_size` for better context
- Check your document is readable (not scanned images)

**4. "Graph is empty"**

Lower the `min_importance` threshold:
```bash
python knowledge_graph.py document.pdf --min-importance 1
```

**5. Rate limiting errors**

Increase `delay_between_batches` in config.yaml:
```yaml
document_processing:
  delay_between_batches: 5  # Increase delay
```

### Debug Mode

Enable verbose logging to see detailed information:

```bash
python knowledge_graph.py document.pdf --verbose
```

## 📚 Project Structure

```
Knowledge_graph/
├── src/
│   ├── config.py              # Configuration management
│   ├── models/                # LLM client implementations
│   │   ├── base.py           # Base LLM interface
│   │   ├── openai_client.py  # OpenAI implementation
│   │   ├── anthropic_client.py # Anthropic implementation
│   │   ├── ollama_client.py  # Ollama implementation
│   │   └── factory.py        # LLM factory
│   ├── extractors/           # Knowledge extraction
│   │   ├── concept_extractor.py # Concept extraction
│   │   └── graph_builder.py  # Graph building
│   └── utils/                # Utilities
│       └── document_processor.py # Document processing
├── examples/                  # Example files
├── data/                     # Data directory
│   └── input/               # Input documents
├── output/                   # Generated outputs
├── logs/                     # Log files
├── knowledge_graph.py        # Main CLI application
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Knowledge_graph.git
cd Knowledge_graph

# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) for document processing
- Powered by [NetworkX](https://networkx.org/) for graph analysis
- Visualizations created with [PyVis](https://pyvis.readthedocs.io/)
- Supports [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), and [Ollama](https://ollama.ai/)

## 📧 Contact & Support

- Issues: [GitHub Issues](https://github.com/yourusername/Knowledge_graph/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/Knowledge_graph/discussions)

## 🗺️ Roadmap

- [ ] Support for more document formats (DOCX, HTML, Markdown)
- [ ] Multi-document knowledge graph merging
- [ ] Export to graph databases (Neo4j, etc.)
- [ ] Web interface for easier usage
- [ ] Incremental graph updates
- [ ] Support for more LLM providers (Cohere, HuggingFace, etc.)

---

**Happy Knowledge Graphing! 🎉**

If you find this project useful, please consider giving it a ⭐ on GitHub!
