# LanceDB Node.js Vector Search

A Node.js implementation for vector search using LanceDB and Ollama's embedding model.

## Overview

This project demonstrates how to:
- Connect to a LanceDB database
- Create custom embedding functions using Ollama
- Perform vector similarity search against stored documents
- Process and display search results

## Prerequisites

- Node.js (v14 or later)
- Ollama running locally with the `nomic-embed-text` model
- LanceDB storage location with read/write permissions

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pnpm install
```

## Dependencies

- `@lancedb/lancedb`: LanceDB client for Node.js
- `apache-arrow`: For handling columnar data
- `node-fetch`: For making API calls to Ollama

## Usage

Run the vector search test script:

```bash
pnpm test-vector-search
```

Or directly execute:

```bash
node test-vector-search.js
```

## Configuration

The script connects to:
- LanceDB at the configured path
- Ollama API at `http://localhost:11434/api/embeddings`

## MCP Configuration

To integrate with Claude Desktop as an MCP service, add the following to your MCP configuration JSON:

```json
{
  "mcpServers": {
    "lanceDB": {
      "command": "node",
      "args": [
        "/path/to/lancedb-node/dist/index.js",
        "--db-path",
        "/path/to/your/lancedb/storage"
      ]
    }
  }
}
```

Replace the paths with your actual installation paths:
- `/path/to/lancedb-node/dist/index.js` - Path to the compiled index.js file
- `/path/to/your/lancedb/storage` - Path to your LanceDB storage directory

## Custom Embedding Function

The project includes a custom `OllamaEmbeddingFunction` that:
- Sends text to the Ollama API
- Receives embeddings with 768 dimensions
- Formats them for use with LanceDB

## Vector Search Example

The example searches for "how to define success criteria" in the "ai-rag" table, displaying results with their similarity scores.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.