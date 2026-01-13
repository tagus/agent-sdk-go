package openai2

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/tagus/agent-sdk-go/pkg/interfaces"
	"github.com/tagus/agent-sdk-go/pkg/llm"
	"github.com/tagus/agent-sdk-go/pkg/logging"
)

type mockTransport struct {
	handler func(*http.Request) (*http.Response, error)
}

func (m mockTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return m.handler(r)
}

func newMockClient(t *testing.T, model string, handler func(w http.ResponseWriter, r *http.Request)) *OpenAIClient {
	t.Helper()

	httpHandler := func(r *http.Request) (*http.Response, error) {
		recorder := httptest.NewRecorder()
		handler(recorder, r)
		return recorder.Result(), nil
	}

	httpClient := &http.Client{Transport: mockTransport{handler: httpHandler}}

	opts := []option.RequestOption{
		option.WithAPIKey("test-key"),
		option.WithBaseURL("http://example.com/"),
		option.WithHTTPClient(httpClient),
	}

	client := NewClient("test-key",
		WithModel(model),
		WithLogger(logging.New()),
	)

	client.Client = openai.NewClient(opts...)
	client.ChatService = openai.NewChatService(opts...)
	client.ResponseService = openai.NewClient(opts...).Responses

	return client
}

func TestGenerate(t *testing.T) {
	client := newMockClient(t, "gpt-4", func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Expected Authorization header with test-key")
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: "test response",
						Role:    "assistant",
					},
				},
			},
		}
		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	})

	// Test generation
	resp, err := client.Generate(context.Background(), "test prompt")
	if err != nil {
		t.Fatalf("Failed to generate: %v", err)
	}

	if resp != "test response" {
		t.Errorf("Expected response 'test response', got '%s'", resp)
	}
}

func TestChat(t *testing.T) {
	client := newMockClient(t, "gpt-4", func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: "test response",
						Role:    "assistant",
					},
				},
			},
		}
		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	})

	// Test chat
	messages := []llm.Message{
		{
			Role:    "user",
			Content: "test message",
		},
	}

	resp, err := client.Chat(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("Failed to chat: %v", err)
	}

	if resp != "test response" {
		t.Errorf("Expected response 'test response', got '%s'", resp)
	}
}

func TestGenerateWithResponseFormat(t *testing.T) {
	client := newMockClient(t, "gpt-4", func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Verify response format is present
		if reqBody["response_format"] == nil {
			t.Error("Expected response_format in request")
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: `{"name": "test", "value": 123}`,
						Role:    "assistant",
					},
				},
			},
		}
		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	})

	// Test generation with response format
	resp, err := client.Generate(context.Background(), "test prompt",
		WithResponseFormat(interfaces.ResponseFormat{
			Name: "test_format",
			Schema: interfaces.JSONSchema{
				"type": "object",
				"properties": map[string]interface{}{
					"name":  map[string]interface{}{"type": "string"},
					"value": map[string]interface{}{"type": "number"},
				},
			},
		}),
	)
	if err != nil {
		t.Fatalf("Failed to generate: %v", err)
	}

	expected := `{"name": "test", "value": 123}`
	if resp != expected {
		t.Errorf("Expected response '%s', got '%s'", expected, resp)
	}
}

func TestChatWithToolMessages(t *testing.T) {
	client := newMockClient(t, "gpt-4", func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Verify that tool messages are present with tool_call_id
		messages := reqBody["messages"].([]interface{})
		foundToolMessage := false
		for _, msg := range messages {
			msgMap := msg.(map[string]interface{})
			if msgMap["role"] == "tool" {
				foundToolMessage = true
				if msgMap["tool_call_id"] != "test-tool-call-id" {
					t.Errorf("Expected tool_call_id 'test-tool-call-id', got '%s'", msgMap["tool_call_id"])
				}
				break
			}
		}
		if !foundToolMessage {
			t.Error("Expected tool message in request")
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		response := openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Content: "test response",
						Role:    "assistant",
					},
				},
			},
		}
		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	})

	// Test chat with tool messages
	messages := []llm.Message{
		{
			Role:    "user",
			Content: "test message",
		},
		{
			Role:       "tool",
			Content:    "tool result",
			ToolCallID: "test-tool-call-id",
		},
	}

	resp, err := client.Chat(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("Failed to chat: %v", err)
	}

	if resp != "test response" {
		t.Errorf("Expected response 'test response', got '%s'", resp)
	}
}

func TestParallelToolExecution(t *testing.T) {
	client := newMockClient(t, "gpt-4", func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}

		// Parse request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Check if this is the first request (with tools) or second request (with tool results)
		messages := reqBody["messages"].([]interface{})
		hasToolResults := false
		for _, msg := range messages {
			msgMap := msg.(map[string]interface{})
			if msgMap["role"] == "tool" {
				hasToolResults = true
				break
			}
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		var response openai.ChatCompletion

		if !hasToolResults {
			// First request - return tool calls
			response = openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: "",
							Role:    "assistant",
							ToolCalls: []openai.ChatCompletionMessageToolCallUnion{
								{
									ID: "call_123",
									Function: openai.ChatCompletionMessageFunctionToolCallFunction{
										Name: "parallel_tool_use",
										Arguments: `{
											"tool_uses": [
												{
													"recipient_name": "test_tool_1",
													"parameters": {"param1": "value1"}
												},
												{
													"recipient_name": "test_tool_2",
													"parameters": {"param2": "value2"}
												}
											]
										}`,
									},
								},
							},
						},
					},
				},
			}
		} else {
			// Second request - return final response
			response = openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: "Final response after parallel tools",
							Role:    "assistant",
						},
					},
				},
			}
		}

		err := json.NewEncoder(w).Encode(response)
		if err != nil {
			t.Fatalf("Failed to encode response: %v", err)
		}
	})

	// Create mock tools
	mockTools := []interfaces.Tool{
		&mockTool{name: "test_tool_1", description: "Test tool 1"},
		&mockTool{name: "test_tool_2", description: "Test tool 2"},
	}

	// Test parallel tool execution
	resp, err := client.GenerateWithTools(context.Background(), "test prompt", mockTools)
	if err != nil {
		t.Fatalf("Failed to generate with tools: %v", err)
	}

	expected := "Final response after parallel tools"
	if resp != expected {
		t.Errorf("Expected response '%s', got '%s'", expected, resp)
	}
}

// mockTool implements interfaces.Tool for testing
type mockTool struct {
	name        string
	description string
}

func (m *mockTool) Name() string {
	return m.name
}

func (m *mockTool) DisplayName() string {
	return m.name
}

func (m *mockTool) Description() string {
	return m.description
}

func (m *mockTool) Internal() bool {
	return false
}

func (m *mockTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"param": {
			Type:        "string",
			Description: "Test parameter",
			Required:    true,
		},
	}
}

func (m *mockTool) Execute(ctx context.Context, args string) (string, error) {
	return fmt.Sprintf("Result from %s: %s", m.name, args), nil
}

func (m *mockTool) Run(ctx context.Context, input string) (string, error) {
	return m.Execute(ctx, input)
}

func TestReasoningEffort(t *testing.T) {
	client := newMockClient(t, "gpt-5-mini", func(w http.ResponseWriter, r *http.Request) {
		// Parse request body
		var reqBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("Failed to decode request body: %v", err)
		}

		// Verify reasoning_effort is present
		if reqBody["reasoning_effort"] != "low" {
			t.Errorf("Expected reasoning_effort 'low', got '%v'", reqBody["reasoning_effort"])
		}

		// Send response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletion{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Content: "test", Role: "assistant"}},
			},
		})
	})

	// Test with reasoning effort
	_, err := client.Generate(context.Background(), "test",
		WithReasoning("low"),
	)
	if err != nil {
		t.Fatalf("Failed to generate: %v", err)
	}
}

func TestGenerateWithMemory(t *testing.T) {
	tests := []struct {
		name     string
		history  []interfaces.Message
		prompt   string
		expected int // expected number of messages in request
	}{
		{
			name:     "empty memory",
			history:  nil, // No memory provided
			prompt:   "Hello",
			expected: 1, // Just the current user message
		},
		{
			name: "conversation with system message",
			history: []interfaces.Message{
				{Role: interfaces.MessageRoleSystem, Content: "You are helpful"},
				{Role: interfaces.MessageRoleUser, Content: "Hi"},
				{Role: interfaces.MessageRoleAssistant, Content: "Hello!"},
				{Role: interfaces.MessageRoleUser, Content: "How are you?"}, // Current prompt should be in memory
			},
			prompt:   "How are you?",
			expected: 4, // system + user + assistant + current user (from memory)
		},
		{
			name: "conversation with tool call",
			history: []interfaces.Message{
				{Role: interfaces.MessageRoleUser, Content: "Check status"},
				{Role: interfaces.MessageRoleAssistant, Content: "Checking..."},
				{
					Role:       interfaces.MessageRoleTool,
					Content:    "All good",
					ToolCallID: "call_123",
					Metadata:   map[string]interface{}{"tool_name": "status_check"},
				},
				{Role: interfaces.MessageRoleUser, Content: "Thanks"}, // Current prompt should be in memory
			},
			prompt:   "Thanks",
			expected: 4, // user + assistant + tool + current user (from memory)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClient(t, "gpt-4o-mini", func(w http.ResponseWriter, r *http.Request) {
				// Parse request body to validate messages
				var reqBody map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
					t.Fatalf("Failed to decode request body: %v", err)
				}

				// Verify messages array
				messages, ok := reqBody["messages"].([]interface{})
				if !ok {
					t.Fatalf("Expected messages array in request")
				}

				if len(messages) != tt.expected {
					t.Errorf("Expected %d messages in request, got %d", tt.expected, len(messages))
				}

				// Verify system message comes first if present
				if len(tt.history) > 0 {
					hasSystemMessage := false
					for _, msg := range tt.history {
						if msg.Role == interfaces.MessageRoleSystem {
							hasSystemMessage = true
							break
						}
					}

					if hasSystemMessage && len(messages) > 0 {
						firstMsg := messages[0].(map[string]interface{})
						if firstMsg["role"] != "system" {
							t.Errorf("Expected first message to be system message, got: %v", firstMsg["role"])
						}
					}
				}

				// Send mock response
				w.Header().Set("Content-Type", "application/json")
				response := openai.ChatCompletion{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Content: "test response",
								Role:    "assistant",
							},
						},
					},
				}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Fatalf("Failed to encode response: %v", err)
				}
			})

			var memory interfaces.Memory
			if tt.history != nil {
				memory = &mockMemory{messages: tt.history}
			}

			// Test Generate with memory
			_, err := client.Generate(context.Background(), tt.prompt,
				interfaces.WithMemory(memory))

			if err != nil {
				t.Fatalf("Generate failed: %v", err)
			}
		})
	}
}
