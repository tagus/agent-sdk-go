# Vector Store

This document explains how to use the Vector Store component of the Agent SDK.

## Overview

Vector stores are used to store and retrieve vector embeddings, which are numerical representations of text that capture semantic meaning. They enable semantic search and retrieval of information based on similarity.

## Supported Vector Stores

### Pinecone

```go
import (
    "github.com/tagus/agent-sdk-go/pkg/vectorstore/pinecone"
    "github.com/tagus/agent-sdk-go/pkg/config"
)

// Get configuration
cfg := config.Get()

// Create Pinecone vector store
store := pinecone.New(
    cfg.VectorStore.Pinecone.APIKey,
    cfg.VectorStore.Pinecone.Environment,
    cfg.VectorStore.Pinecone.Index,
)
```

## Using Vector Stores

### Adding Documents

Add documents to the vector store:

```go
import (
    "context"
    "github.com/tagus/agent-sdk-go/pkg/interfaces"
)

// Create documents
docs := []interfaces.Document{
    {
        ID:      "doc1",
        Content: "This is the first document about artificial intelligence.",
        Metadata: map[string]interface{}{
            "source": "article",
            "author": "John Doe",
        },
    },
    {
        ID:      "doc2",
        Content: "This is the second document about machine learning.",
        Metadata: map[string]interface{}{
            "source": "book",
            "author": "Jane Smith",
        },
    },
}

// Add documents to the vector store
err := store.AddDocuments(context.Background(), docs)
if err != nil {
    log.Fatalf("Failed to add documents: %v", err)
}
```

### Searching Documents

Search for documents by similarity:

```go
// Search for documents similar to a query
results, err := store.Search(
    context.Background(),
    "What is artificial intelligence?",
    interfaces.WithLimit(5),
)
if err != nil {
    log.Fatalf("Failed to search documents: %v", err)
}

// Print search results
for _, result := range results {
    fmt.Printf("Document ID: %s\n", result.ID)
    fmt.Printf("Content: %s\n", result.Content)
    fmt.Printf("Score: %f\n", result.Score)
    fmt.Println("Metadata:", result.Metadata)
    fmt.Println()
}
```

### Retrieving Documents

Retrieve documents by ID:

```go
// Retrieve documents by ID
docs, err := store.GetDocuments(
    context.Background(),
    []string{"doc1", "doc2"},
)
if err != nil {
    log.Fatalf("Failed to retrieve documents: %v", err)
}

// Print retrieved documents
for _, doc := range docs {
    fmt.Printf("Document ID: %s\n", doc.ID)
    fmt.Printf("Content: %s\n", doc.Content)
    fmt.Println("Metadata:", doc.Metadata)
    fmt.Println()
}
```

### Deleting Documents

Delete documents from the vector store:

```go
// Delete documents by ID
err := store.DeleteDocuments(
    context.Background(),
    []string{"doc1"},
)
if err != nil {
    log.Fatalf("Failed to delete documents: %v", err)
}
```

## Configuration Options

### Pinecone Options

```go
// Set the namespace
pinecone.WithNamespace("default")

// Set the organization ID for multi-tenancy
pinecone.WithOrgID("org-123")
```

## Multi-tenancy with Vector Stores

When using vector stores with multi-tenancy, you can specify the organization ID:

```go
import (
    "context"
    "github.com/tagus/agent-sdk-go/pkg/multitenancy"
)

// Create context with organization ID
ctx := context.Background()
ctx = multitenancy.WithOrgID(ctx, "org-123")

// Add documents for this organization
err := store.AddDocuments(ctx, docs)

// Search documents for this organization
results, err := store.Search(ctx, "artificial intelligence")
```

## Creating Custom Vector Store Implementations

You can implement custom vector stores by implementing the `interfaces.VectorStore` interface:

```go
import (
    "context"
    "github.com/tagus/agent-sdk-go/pkg/interfaces"
)

// CustomVectorStore is a custom vector store implementation
type CustomVectorStore struct {
    // Add your fields here
}

// NewCustomVectorStore creates a new custom vector store
func NewCustomVectorStore() *CustomVectorStore {
    return &CustomVectorStore{}
}

// AddDocuments adds documents to the vector store
func (s *CustomVectorStore) AddDocuments(ctx context.Context, documents []interfaces.Document) error {
    // Implement your logic to add documents
    return nil
}

// Search searches for documents similar to the query
func (s *CustomVectorStore) Search(ctx context.Context, query string, options ...interfaces.SearchOption) ([]interfaces.SearchResult, error) {
    // Apply options
    opts := &interfaces.SearchOptions{}
    for _, option := range options {
        option(opts)
    }

    // Implement your search logic
    return []interfaces.SearchResult{
        {
            ID:      "doc1",
            Content: "This is a document about artificial intelligence.",
            Score:   0.95,
            Metadata: map[string]interface{}{
                "source": "article",
            },
        },
    }, nil
}

// GetDocuments retrieves documents by ID
func (s *CustomVectorStore) GetDocuments(ctx context.Context, ids []string) ([]interfaces.Document, error) {
    // Implement your logic to retrieve documents
    return []interfaces.Document{
        {
            ID:      "doc1",
            Content: "This is a document about artificial intelligence.",
            Metadata: map[string]interface{}{
                "source": "article",
            },
        },
    }, nil
}

// DeleteDocuments deletes documents by ID
func (s *CustomVectorStore) DeleteDocuments(ctx context.Context, ids []string) error {
    // Implement your logic to delete documents
    return nil
}

// Name returns the name of the vector store
func (s *CustomVectorStore) Name() string {
    return "custom-vector-store"
}
```

## Using Vector Stores with Embeddings

