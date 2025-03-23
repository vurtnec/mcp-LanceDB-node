#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, ToolSchema, } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import * as lancedb from "@lancedb/lancedb";
import fetch from "node-fetch";
// Parse command line arguments
const args = process.argv.slice(2);
let dbPath;
let ollamaEndpoint = "http://localhost:11434/api/embeddings";
let ollamaModel = "nomic-embed-text:latest";
let showHelp = false;
for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
        case '--db-path':
            dbPath = args[++i];
            break;
        case '--ollama-endpoint':
            ollamaEndpoint = args[++i];
            break;
        case '--ollama-model':
            ollamaModel = args[++i];
            break;
        case '--help':
        case '-h':
            showHelp = true;
            break;
    }
}
// Show help message if requested
if (showHelp) {
    console.error('Usage: mcp-server-lancedb --db-path <path> [--ollama-endpoint <url>] [--ollama-model <model>]');
    console.error('');
    console.error('Options:');
    console.error('  --db-path <path>           Path to the LanceDB database (required)');
    console.error('  --ollama-endpoint <url>    URL of the Ollama API embeddings endpoint (default: http://localhost:11434/api/embeddings)');
    console.error('  --ollama-model <model>     Ollama model to use for embeddings (default: nomic-embed-text:latest)');
    console.error('  --help, -h                 Show this help message');
    process.exit(0);
}
// Validate required command line arguments
if (!dbPath) {
    console.error('Error: Missing required arguments');
    console.error('Usage: mcp-server-lancedb --db-path <path> [--ollama-endpoint <url>] [--ollama-model <model>]');
    process.exit(1);
}
// Ollama API调用函数
async function generateEmbedding(text) {
    try {
        const response = await fetch(ollamaEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: ollamaModel,
                prompt: text
            }),
        });
        if (!response.ok) {
            throw new Error(`Ollama API错误: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        return data.embedding;
    }
    catch (error) {
        console.error(`生成嵌入时出错: ${error.message}`);
        throw error;
    }
}
// Schema definitions
const VectorAddArgsSchema = z.object({
    table_name: z.string().describe('Name of the table to add vectors to'),
    vectors: z.array(z.object({
        vector: z.array(z.number()).describe('Vector data'),
    }).passthrough()).describe('Array of vectors with metadata to add')
});
const VectorSearchArgsSchema = z.object({
    table_name: z.string().describe('Name of the table to search in'),
    query_vector: z.array(z.number()).optional().describe('Query vector for similarity search'),
    query_text: z.string().optional().describe('Text query to be converted to a vector using Ollama embedding'),
    limit: z.number().optional().describe('Maximum number of results to return (default: 10)'),
    distance_type: z.enum(['l2', 'cosine', 'dot']).optional().describe('Distance metric to use (default: cosine)'),
    where: z.string().optional().describe('Filter condition in SQL syntax'),
    with_vectors: z.boolean().optional().describe('Whether to include vector data in results (default: false)')
}).refine((data) => {
    return data.query_vector !== undefined || data.query_text !== undefined;
}, {
    message: "Either query_vector or query_text must be provided"
});
const ListTablesArgsSchema = z.object({});
const ToolInputSchema = ToolSchema.shape.inputSchema;
// Connect to the LanceDB database
let db;
async function connectDB() {
    try {
        db = await lancedb.connect(dbPath);
        console.error(`Connected to LanceDB at: ${dbPath}`);
    }
    catch (error) {
        console.error(`Failed to connect to LanceDB: ${error}`);
        process.exit(1);
    }
}
// Server setup
const server = new Server({
    name: "lancedb-vector-server",
    version: "0.1.0",
}, {
    capabilities: {
        tools: {},
    },
});
// Tool implementations
// Tool handlers
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: "vector_add",
                description: "Add vectors with metadata to a LanceDB table. Creates the table if it doesn't exist.",
                inputSchema: zodToJsonSchema(VectorAddArgsSchema),
            },
            {
                name: "vector_search",
                description: "Search for similar vectors in a LanceDB table using either a direct vector or text that will be converted to a vector using Ollama embedding.",
                inputSchema: zodToJsonSchema(VectorSearchArgsSchema),
            },
            {
                name: "list_tables",
                description: "List all tables in the LanceDB database.",
                inputSchema: zodToJsonSchema(ListTablesArgsSchema),
            }
        ]
    };
});
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
        const { name, arguments: args } = request.params;
        switch (name) {
            case "vector_add": {
                const parsed = VectorAddArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for vector_add: ${parsed.error}`);
                }
                const { table_name, vectors } = parsed.data;
                // Check if table exists, if not create it
                try {
                    const table = await db.openTable(table_name);
                    await table.add(vectors);
                    return {
                        content: [{ type: "text", text: `Added ${vectors.length} vectors to table ${table_name}` }],
                        isError: false,
                    };
                }
                catch (error) {
                    // Table doesn't exist, create it
                    if (error.message.includes("does not exist")) {
                        const table = await db.createTable(table_name, vectors);
                        return {
                            content: [{ type: "text", text: `Created table ${table_name} and added ${vectors.length} vectors` }],
                            isError: false,
                        };
                    }
                    else {
                        throw error;
                    }
                }
            }
            case "vector_search": {
                const parsed = VectorSearchArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for vector_search: ${parsed.error}`);
                }
                const { table_name, query_vector, query_text, limit = 10, distance_type = 'cosine', where, with_vectors = false } = parsed.data;
                try {
                    const table = await db.openTable(table_name);
                    // 确定使用哪个查询向量
                    let searchVector;
                    if (query_vector) {
                        // 直接使用提供的向量
                        searchVector = query_vector;
                    }
                    else if (query_text) {
                        // 使用Ollama生成文本的嵌入向量
                        console.error(`Generating embedding for text: "${query_text}"`);
                        searchVector = await generateEmbedding(query_text);
                        console.error(`Generated embedding with dimension: ${searchVector.length}`);
                    }
                    else {
                        throw new Error('Either query_vector or query_text must be provided');
                    }
                    // 检查表格结构和向量维度
                    const schema = await table.schema();
                    const vectorField = schema.fields.find(field => field.type.typeId === 16 && field.type.listSize > 10);
                    if (!vectorField) {
                        throw new Error('No vector column found in the table');
                    }
                    const vectorColumnName = vectorField.name;
                    const vectorDimension = vectorField.type.listSize;
                    if (searchVector.length !== vectorDimension) {
                        console.error(`Warning: Query vector dimension (${searchVector.length}) doesn't match table vector dimension (${vectorDimension})`);
                    }
                    // 执行向量搜索
                    let query = table.vectorSearch(searchVector);
                    // 设置距离类型和限制
                    query = query.distanceType(distance_type).limit(limit);
                    if (where) {
                        query = query.where(where);
                    }
                    const originalResults = await query.toArray();
                    // 创建新的结果对象，而不是修改原始对象
                    let processedResults;
                    if (!with_vectors) {
                        processedResults = originalResults.map(item => {
                            // 创建新对象，排除vector属性
                            const itemCopy = { ...item };
                            delete itemCopy[vectorColumnName];
                            return itemCopy;
                        });
                    }
                    else {
                        processedResults = originalResults;
                    }
                    return {
                        content: [{ type: "text", text: JSON.stringify(processedResults, null, 2) }],
                        isError: false,
                    };
                }
                catch (error) {
                    throw new Error(`Error searching table ${table_name}: ${error}`);
                }
            }
            case "list_tables": {
                try {
                    const tables = await db.tableNames();
                    return {
                        content: [{ type: "text", text: JSON.stringify(tables, null, 2) }],
                        isError: false,
                    };
                }
                catch (error) {
                    throw new Error(`Error listing tables: ${error}`);
                }
            }
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
            content: [{ type: "text", text: `Error: ${errorMessage}` }],
            isError: true,
        };
    }
});
// Start server
async function runServer() {
    await connectDB();
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("LanceDB MCP Server running on stdio");
}
runServer().catch((error) => {
    console.error("Fatal error running server:", error);
    process.exit(1);
});
