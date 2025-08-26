#!/bin/bash
set -e

# Run-all script for MCP RAG application
# Starts PDF Tool, RAG Tool, Host App, and MCP Server

echo "Starting PDF Tool Server on port 8001..."
uvicorn pdf_tool_server:app --host 0.0.0.0 --port 8001 &
pids+=("$!")

echo "Starting RAG Tool Server on port 8002..."
uvicorn rag_tool_server:app --host 0.0.0.0 --port 8002 &
pids+=("$!")

echo "Starting Host App on port 8000..."
uvicorn host_app:app --host 0.0.0.0 --port 8000 &
pids+=("$!")

echo "Starting MCP Server on port 8003..."
uvicorn mcp_server:app --host 0.0.0.0 --port 8003 &
pids+=("$!")

echo "All services started (PIDs: ${pids[*]})"

# Trap SIGINT/SIGTERM to shutdown all
trap "echo 'Shutting down...'; kill ${pids[*]}; exit" SIGINT SIGTERM

# Wait for any process to exit
wait -n

# Kill remaining processes
echo "One process exited, killing others..."
kill ${pids[*]}
echo "All services stopped."
