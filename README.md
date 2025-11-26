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


This project extends the base Helix Navigator agent with an internal conversation workflow history component. The goal of this addition is to track the sequence of operations the LangGraph workflow performs for each user query, creating a transparent record of recent interactions.
The history module maintains a limited rolling window of recent conversation turns (default 10). For each turn, the agent stores the user question along with the intermediate outputs produced during the workflow—question classification, entity extraction, query generation, database execution summary, and the final formatted answer. These values are captured automatically through small hooks placed in each workflow node.

**State Management:**
- The history module functions as a dedicated component attached to the workflow, maintaining a structured record of recent conversation turns. It operates independently of the main agent logic while remaining accessible to any step in the LangGraph workflow.
**Decoupled Capture:**
- Lightweight hook functions are embedded within each workflow node to record intermediate outputs (classification, extracted entities, generated query, execution summary, etc.). These hooks are intentionally non-intrusive—capturing information without altering the state transitions or control flow of the agent.
**Rolling Window Memory:**
- The system maintains a bounded history using a configurable FIFO buffer (default: 10 turns). When the buffer reaches capacity, the oldest turn is automatically removed, keeping memory usage stable and ensuring the history remains efficient and manageable.

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
├── src/agents/              # AI agent implementations
├── src/web/app.py          # Interactive Streamlit interface
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

