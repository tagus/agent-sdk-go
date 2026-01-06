#!/bin/bash

# Generate gRPC Go code from protobuf definitions

set -e

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "Error: protoc is not installed. Please install Protocol Buffer Compiler."
    echo "On macOS: brew install protobuf"
    echo "On Ubuntu: sudo apt-get install protobuf-compiler"
    exit 1
fi

# Check if protoc-gen-go is installed
if ! command -v protoc-gen-go &> /dev/null; then
    echo "Installing protoc-gen-go..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

# Check if protoc-gen-go-grpc is installed
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "Installing protoc-gen-go-grpc..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

# Create output directory
mkdir -p pkg/grpc/pb

# Generate Go code from proto files
echo "Generating gRPC Go code..."
protoc \
    --proto_path=pkg/grpc/proto \
    --go_out=pkg/grpc/pb \
    --go_opt=paths=source_relative \
    --go-grpc_out=pkg/grpc/pb \
    --go-grpc_opt=paths=source_relative \
    agent.proto

echo "gRPC Go code generated successfully!"
