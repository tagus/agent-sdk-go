package openai2

import (
	"context"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/responses"
	"github.com/tagus/agent-sdk-go/pkg/interfaces"
	"github.com/tagus/agent-sdk-go/pkg/logging"
)

// messageHistoryBuilder builds OpenAI-compatible message history from memory and current prompt
type messageHistoryBuilder struct {
	logger logging.Logger
}

// newMessageHistoryBuilder creates a new message history builder
func newMessageHistoryBuilder(logger logging.Logger) *messageHistoryBuilder {
	return &messageHistoryBuilder{
		logger: logger,
	}
}

// buildMessages constructs OpenAI messages from memory and current prompt
// Returns messages ready for OpenAI API calls, preserving chronological order
func (b *messageHistoryBuilder) buildMessages(ctx context.Context, prompt string, memory interfaces.Memory) []openai.ChatCompletionMessageParamUnion {
	messages := []openai.ChatCompletionMessageParamUnion{}

	// Add memory messages
	if memory != nil {
		memoryMessages, err := memory.GetMessages(ctx)
		if err != nil {
			b.logger.Error(ctx, "Failed to retrieve memory messages", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			// Convert memory messages to OpenAI format, preserving chronological order
			for _, msg := range memoryMessages {
				openaiMsg := b.convertMemoryMessage(msg)
				if openaiMsg != nil {
					messages = append(messages, *openaiMsg)
				}
			}
		}
	} else {
		// Only append current user message when memory is nil
		messages = append(messages, openai.UserMessage(prompt))
	}

	return messages
}

// buildResponseInputItems constructs Response API input items from memory and current prompt
func (b *messageHistoryBuilder) buildResponseInputItems(ctx context.Context, prompt string, memory interfaces.Memory) []responses.ResponseInputItemUnionParam {
	items := []responses.ResponseInputItemUnionParam{}

	// Add memory messages if present
	if memory != nil {
		memoryMessages, err := memory.GetMessages(ctx)
		if err != nil {
			b.logger.Error(ctx, "Failed to retrieve memory messages", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			for _, msg := range memoryMessages {
				if converted := b.convertMemoryMessageToResponseInput(msg); len(converted) > 0 {
					items = append(items, converted...)
				}
			}
		}
	} else {
		// Only append current user message when memory is nil
		items = append(items, responses.ResponseInputItemParamOfMessage(prompt, responses.EasyInputMessageRoleUser))
	}

	return items
}

// convertMemoryMessage converts a memory message to OpenAI format
func (b *messageHistoryBuilder) convertMemoryMessage(msg interfaces.Message) *openai.ChatCompletionMessageParamUnion {
	switch msg.Role {
	case interfaces.MessageRoleUser:
		userMsg := openai.UserMessage(msg.Content)
		return &userMsg

	case interfaces.MessageRoleAssistant:
		if len(msg.ToolCalls) > 0 {
			// Assistant message with tool calls
			var toolCalls []openai.ChatCompletionMessageToolCallUnion

			for _, toolCall := range msg.ToolCalls {
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnion{
					ID:   toolCall.ID,
					Type: "function",
					Function: openai.ChatCompletionMessageFunctionToolCallFunction{
						Name:      toolCall.Name,
						Arguments: toolCall.Arguments,
					},
				})
			}

			// Create assistant message with tool calls
			assistantMsg := openai.ChatCompletionMessage{
				Role:      "assistant",
				Content:   msg.Content,
				ToolCalls: toolCalls,
			}
			param := assistantMsg.ToParam()
			return &param
		} else if msg.Content != "" {
			// Regular assistant message
			assistantMsg := openai.AssistantMessage(msg.Content)
			return &assistantMsg
		}

	case interfaces.MessageRoleTool:
		if msg.ToolCallID != "" {
			toolMsg := openai.ToolMessage(msg.Content, msg.ToolCallID)
			return &toolMsg
		}

	case interfaces.MessageRoleSystem:
		// Convert system messages from memory to OpenAI system messages
		systemMsg := openai.SystemMessage(msg.Content)
		return &systemMsg
	}

	return nil
}

// convertMemoryMessageToResponseInput converts a memory message to Responses API input items
func (b *messageHistoryBuilder) convertMemoryMessageToResponseInput(msg interfaces.Message) []responses.ResponseInputItemUnionParam {
	switch msg.Role {
	case interfaces.MessageRoleUser:
		item := responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleUser)
		return []responses.ResponseInputItemUnionParam{item}
	case interfaces.MessageRoleAssistant:
		items := []responses.ResponseInputItemUnionParam{}

		// Include assistant text when present
		if msg.Content != "" {
			items = append(items, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleAssistant))
		}

		// Add tool calls as function call items
		for _, toolCall := range msg.ToolCalls {
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(toolCall.Arguments, toolCall.ID, toolCall.Name))
		}

		return items
	case interfaces.MessageRoleTool:
		if msg.ToolCallID != "" {
			item := responses.ResponseInputItemParamOfFunctionCallOutput(msg.ToolCallID, msg.Content)
			return []responses.ResponseInputItemUnionParam{item}
		}
	case interfaces.MessageRoleSystem:
		item := responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleSystem)
		return []responses.ResponseInputItemUnionParam{item}
	}

	return nil
}
