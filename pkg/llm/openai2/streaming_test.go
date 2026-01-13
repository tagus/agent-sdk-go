package openai2

import (
	"bytes"
	"context"
	"io"
	"net/http"
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
