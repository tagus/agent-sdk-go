package openai2

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/responses"
	"github.com/tagus/agent-sdk-go/pkg/interfaces"
	"github.com/tagus/agent-sdk-go/pkg/logging"
)

type mockStreamTransport struct {
	body string
}

func (m mockStreamTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewBufferString(m.body)),
		Header: http.Header{
			"content-type": []string{"text/event-stream"},
		},
	}
	return resp, nil
}

func TestGenerateStreamEmitsThinkingAndUsage(t *testing.T) {
	// Stream two deltas (text + reasoning) and a completed event with usage.
	streamBody := "" +
		"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hi\"}\n\n" +
		"data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Thinking\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"output_tokens_details\":{\"reasoning_tokens\":2},\"total_tokens\":15,\"input_tokens_details\":{\"cached_tokens\":1}}}}\n\n"

	opts := []option.RequestOption{
		option.WithAPIKey("test"),
		option.WithBaseURL("http://example.com/"),
		option.WithHTTPClient(&http.Client{Transport: mockStreamTransport{body: streamBody}}),
	}

	client := &OpenAIClient{
		ResponseService: responses.NewResponseService(opts...),
		Model:           "gpt-4o-mini",
		logger:          logging.New(),
	}

	events, err := client.GenerateStream(context.Background(), "hi")
	if err != nil {
		t.Fatalf("GenerateStream returned error: %v", err)
	}

	var gotThinking bool
	var gotContent bool
	var gotUsage bool

	for ev := range events {
		switch ev.Type {
		case interfaces.StreamEventThinking:
			gotThinking = true
		case interfaces.StreamEventContentDelta:
			if ev.Content == "Hi" {
				gotContent = true
			}
		case interfaces.StreamEventMessageStop:
			if ev.Metadata != nil && ev.Metadata["usage"] != nil {
				gotUsage = true
			}
		case interfaces.StreamEventError:
			t.Fatalf("unexpected error event: %v", ev.Error)
		}
	}

	if !gotContent {
		t.Fatalf("expected content delta event")
	}
	if !gotThinking {
		t.Fatalf("expected thinking event from reasoning delta")
	}
	if !gotUsage {
		t.Fatalf("expected usage metadata on message stop")
	}
}

type sequencedTransport struct {
	bodies   []string
	requests [][]byte
	mu       sync.Mutex
}

func (s *sequencedTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	bodyBytes, _ := io.ReadAll(r.Body)
	r.Body.Close()

	s.mu.Lock()
	s.requests = append(s.requests, bodyBytes)
	idx := len(s.requests) - 1
	if idx >= len(s.bodies) {
		idx = len(s.bodies) - 1
	}
	respBody := s.bodies[idx]
	s.mu.Unlock()

	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewBufferString(respBody)),
		Header: http.Header{
			"content-type": []string{"text/event-stream"},
		},
	}
	return resp, nil
}

func TestGenerateWithToolsStream_UsesResponsesAndTools(t *testing.T) {
	// First stream triggers tool call, second returns final content.
	firstStream := "" +
		"data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item1\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"\"},\"output_index\":0}\n\n" +
		"data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"item1\",\"delta\":\"{\\\"msg\\\":\\\"hi\\\"}\",\"output_index\":0}\n\n" +
		// done repeats the same arguments; we should not duplicate them.
		"data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item1\",\"arguments\":\"{\\\"msg\\\":\\\"hi\\\"}\",\"output_index\":0}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":6,\"input_tokens_details\":{\"cached_tokens\":0}}}}\n\n"

	secondStream := "" +
		"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"done\"}\n\n" +
		"data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"content_index\":0}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":6,\"input_tokens_details\":{\"cached_tokens\":0}}}}\n\n"

	transport := &sequencedTransport{
		bodies: []string{firstStream, secondStream},
	}

	opts := []option.RequestOption{
		option.WithAPIKey("test"),
		option.WithBaseURL("http://example.com/"),
		option.WithHTTPClient(&http.Client{Transport: transport}),
	}

	client := &OpenAIClient{
		ResponseService: responses.NewResponseService(opts...),
		Model:           "gpt-4o-mini",
		logger:          logging.New(),
	}

	tool := &mockTool{name: "echo", description: "echo tool"}

	stream, err := client.GenerateWithToolsStream(context.Background(), "hi", []interfaces.Tool{tool})
	if err != nil {
		t.Fatalf("GenerateWithToolsStream returned error: %v", err)
	}

	var sawToolUse, sawToolResult, sawContent bool
	var toolArgs string
	for ev := range stream {
		switch ev.Type {
		case interfaces.StreamEventToolUse:
			sawToolUse = true
			toolArgs = ev.ToolCall.Arguments
		case interfaces.StreamEventToolResult:
			sawToolResult = true
		case interfaces.StreamEventContentDelta:
			if ev.Content == "done" {
				sawContent = true
			}
		case interfaces.StreamEventError:
			t.Fatalf("unexpected error event: %v", ev.Error)
		}
	}

	if !sawToolUse || !sawToolResult || !sawContent {
		t.Fatalf("missing events toolUse=%v toolResult=%v content=%v", sawToolUse, sawToolResult, sawContent)
	}
	if toolArgs != `{"msg":"hi"}` {
		t.Fatalf("unexpected tool args: %q", toolArgs)
	}

	transport.mu.Lock()
	defer transport.mu.Unlock()
	if len(transport.requests) < 2 {
		t.Fatalf("expected two streamed requests, got %d", len(transport.requests))
	}
	if !strings.Contains(string(transport.requests[1]), "function_call_output") {
		t.Fatalf("second request did not include tool result: %s", transport.requests[1])
	}
}
