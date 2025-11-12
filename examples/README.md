# Examples

This directory contains example documents and notebooks demonstrating the knowledge graph generator.

## Contents

- **knowledge_graph(harry_potter).ipynb** - Jupyter notebook example showing knowledge graph generation from Harry Potter text
- **THE BOY WHO LIVED.txt** - Sample text file from Harry Potter for testing

## Running Examples

### Using the example text file:

```bash
# From the project root directory
python knowledge_graph.py "examples/THE BOY WHO LIVED .txt" --output output/harry_potter
```

### Using the sample PDF:

```bash
python knowledge_graph.py data/input/cureus-0015-00000040274.pdf --output output/medical_paper
```

## Try Different Models

```bash
# Using OpenAI
python knowledge_graph.py "examples/THE BOY WHO LIVED .txt" --provider openai --model gpt-4o-mini

# Using Anthropic Claude
python knowledge_graph.py "examples/THE BOY WHO LIVED .txt" --provider anthropic --model claude-3-5-sonnet-20241022

# Using local Ollama model
python knowledge_graph.py "examples/THE BOY WHO LIVED .txt" --provider ollama --model llama3.2
```

## Expected Output

After running, you should see:
- An interactive HTML visualization
- CSV files with concepts and graph edges
- JSON files with graph data and statistics

Open the generated `knowledge_graph.html` file in your browser to explore the interactive graph!
