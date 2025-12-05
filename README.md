# Persistent Memory for Multi-Step Biomedical Reasoning

**Learn LangGraph and Knowledge Graphs through Biomedical AI**

An interactive educational project that teaches modern AI development through hands-on biomedical applications. Build AI agents that answer complex questions about genes, proteins, diseases, and drugs using graph databases and multi-step AI workflows.

*Navigate: [Getting Started](docs/getting-started.md) | [Foundations Guide](docs/foundations-and-background.md) | [Reference](docs/reference.md) | [Technical Guide](docs/technical-guide.md)*

## What You'll Learn

- **Knowledge Graphs**: Represent domain knowledge as nodes and relationships
- **LangGraph**: Build multi-step AI workflows with state management  
- **Cypher Queries**: Query graph databases effectively
- **AI Integration**: Combine language models with structured knowledge
- **Biomedical Applications**: Apply AI to drug discovery and personalized medicine


## Key Contributions

| Feature | Description |
|---------|-------------|
| `WorkflowHistory` class | Sliding window buffer (k=10) storing questions, entities, queries, and classifications |
| Memory-augmented classification | Injects prior question-type pairs to recognize follow-up patterns |
| Pronoun resolution | Injects recent entities into extraction prompts for coreference resolution |
| Query chaining | Injects prior queries into generation prompts to preserve filters |
| Conversation Memory UI | Sidebar panel displaying turn-by-turn history with entities and results |


## Quick Start

1. **New to these concepts?** Read the [Foundations Guide](docs/foundations-and-background.md)
2. **Setup**: Follow [Getting Started](docs/getting-started.md) for installation
3. **Learn**: Use the interactive Streamlit web interface
4. **Practice**: Work through the exercises in the web app

## Technology Stack

- **LangGraph**: AI workflow orchestration
- **Neo4j**: Graph database
- **Anthropic Claude**: Language model
- **Streamlit**: Interactive web interface
- **LangGraph Studio**: Visual debugging

## Installation

**Quick Setup**: Python 3.10+, Neo4j, PDM

```bash
# Install dependencies
pdm install

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Load data and start
pdm run load-data
pdm run app # runs with added memory summary
```

## Project Structure

```
├── src/
│   ├── agents/
│   │   ├── workflow_agent.py      # Main LangGraph agent with memory integration
│   │   ├── workflow_history.py    # WorkflowHistory module
│   │   └── graph_interface.py     # Neo4j database interface
│   └── web/
│       └── app.py                 # Streamlit interface with memory panel
├── docs/                   # Documentation and tutorials
├── data/                   # Biomedical datasets
├── scripts/                # Data loading utilities
└── tests/                  # Test suite
```

**Key Files**:
- `src/agents/workflow_agent.py` - Main LangGraph agent
- `src/agents/workflow_history.py` - Conversation workflow history module added
- `src/web/app.py` - Interactive Streamlit interface
- `docs/` - Complete documentation

## Memory System Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    WorkflowHistory                       │
├─────────────────────────────────────────────────────────┤
│  turns: List[Dict]         # Sliding window buffer      │
│  _current_turn: Dict       # In-progress turn           │
│  max_turns: int = 10       # Buffer capacity            │
├─────────────────────────────────────────────────────────┤
│  start_turn(question)      # Initialize new turn        │
│  add_step(name, content)   # Record workflow step       │
│  set_entities(entities)    # Store extracted entities   │
│  finalize_turn()           # Commit turn to buffer      │
│  get_recent_entities(k)    # Retrieve for pronoun res.  │
│  get_summary(k)            # Retrieve for query chain.  │
│  get_history_summary()     # Format for UI display      │
└─────────────────────────────────────────────────────────┘

## Running the Application

### Basic Usage
```bash
pdm run load-data         # Load biomedical data
pdm run app              # Start web interface with memory addition
```

### Visual Debugging
```bash
pdm run langgraph    # Start LangGraph Studio
```

### Development
```bash
pdm run test            # Run tests (14 tests)
pdm run format          # Format code
pdm run lint            # Check quality
```

**Full commands**: See [Reference Guide](docs/reference.md)

## AI Agent

**WorkflowAgent** - LangGraph implementation with transparent processing for learning core LangGraph concepts through biomedical applications

## Example Questions

- **"Which drugs have high efficacy for treating diseases?"**
- **"Which approved drugs treat cardiovascular diseases?"**
- **"Which genes encode proteins that are biomarkers for diseases?"**
- **"What drugs target proteins with high confidence disease associations?"**
- **"Which approved drugs target specific proteins?"**
- **"Which genes are linked to multiple disease categories?"**
- **"What proteins have causal associations with diseases?"** 

