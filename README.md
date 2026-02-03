# RemNote Graph RAG | AI Practice System

An AI practice and learning system that combines knowledge graph with multi-agent workflows to help you master technical concepts through interactive learning, research, and visualization.

## Overview

This project combines a backend powered by LLM with an interactive Reflex web frontend. It is built on a personal [RemNote](https://www.remnote.com) knowledge base through the creation of a knowledge graph, which is used for querying information and visualizing data. Internal personal knowledge can also be expanded through external web research.

## Architecture

### Backend
The backend is built on a multi-agent workflow architecture powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LlamaIndex](https://developers.llamaindex.ai/python/framework/). The following agents are currently implemented:

- **Orchestrator**: Routes requests to other specialized agents
- **Retriever**: Searches for relevant information in a knowledge graph and performs reranking
- **Researcher**: Utilizes web search via [Tavily](https://www.tavily.com/) API to expand the existing knowledge base
- **Analyst**: Synthesizes and summarizes information
- **Mentor**: Answers questions and facilitates practice
- **Visualizer**: Creates visual representations of knowledge graphs

**Key Technologies:**
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [LlamaIndex](https://developers.llamaindex.ai/python/framework/) for knowledge graph indexing and retrieval
- [Neo4j](https://neo4j.com/) for graph storage
- [Redis](https://redis.io/) for vector and document storage
- [Ollama models](https://ollama.com/library?sort=newest) for local/cloud LLM inference
- [Cohere](https://docs.cohere.com/docs/rerank) for reranking

### Frontend
Web interface built with [Reflex](https://github.com/reflex-dev/reflex) framework:

- Real-time streaming responses
- Interactive graph visualizations with [Plotly](https://plotly.com/)
- Agent status monitoring

## Prerequisites

1. **Python 3.11+**
2. **Poetry** for dependency management
3. **Required Services:**
   - Ollama server (for local LLM inference, but Ollama Cloud can also be used)
   - Neo4j database (for graph storage)
   - Redis server (for vector/document storage)

## Prepare environment
Download RemNote data in Markdown format and move all .md files to the `data.raw.<remnote_data_folder>` directory (specify the <remnote_data_folder> in the `backend.configs.constants.REMNOTE_FOLDER_NAME` variable).\
Next, we need to parse the files: run the `scripts.parse_data.py` script.\
Finally, create the knowledge graph from the parsed data: run the `scripts.build_graph_index.py` script.

Both scripts will save the data to the local storage by default (the path is specified in the `backend.configs.paths.PathSettings.local_storage_dir` parameter). See the comments in the scripts to use non-local storage (Redis and Neo4j).\
Now we are ready to run the application.


## Running the Application

```bash
# Initialize Reflex (first time only)
reflex init

# Start the Reflex development server
reflex run

# For production deployment
reflex run --env prod
```

The application will be available at `http://localhost:3000`

## Usage Examples
- You may choose one of the suggested requests or ask you own question

![Alt text](app/assets/welcome_screen.png)

- Visualize the knowledge from the personal knowledge graph

![Alt text](app/assets/visualization.png)

- Research the topic using the Web

![Alt text](app/assets/research.png)

- Get some quiz

![Alt text](app/assets/mentor.png)


## Future Work
- Update the Knowledge Graph visualization: Chunk nodes are currently named by their IDs, not text names, like Entity nodes
- Consider adding more flexible graph traversal options, perhaps a dynamic one?
- Testing! The main part of the project is missing
- Add a database to store user sessions, logs, and other data. Redis is only for the vector search
- Check [Langsmith](https://smith.langchain.com/) – is it working?
- **[BUG]** Jobs marked with flags after completion (e.g., retrievers, visualization) prevent users from requesting the same job on a different topic in the same chat (e.g. when asking to find the information from personal KG on different topics in one chat). Consider refreshing the state context and turning the visualization context to a list, allowing access to all previous plots while generating multiple ones
- Try [DSPy](https://github.com/stanfordnlp/dspy)?