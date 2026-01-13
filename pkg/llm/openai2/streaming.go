package openai2

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
	"github.com/tagus/agent-sdk-go/pkg/interfaces"
	"github.com/tagus/agent-sdk-go/pkg/multitenancy"
)

// GenerateStream implements interfaces.StreamingLLM.GenerateStream
func (c *OpenAIClient) GenerateStream(
	ctx context.Context,
	prompt string,
	options ...interfaces.GenerateOption,
) (<-chan interfaces.StreamEvent, error) {
	// Apply options
	params := &interfaces.GenerateOptions{
		LLMConfig: &interfaces.LLMConfig{
			Temperature: 0.7,
		},
	}

	for _, option := range options {
		option(params)
	}

	// Check for organization ID in context
	defaultOrgID := "default"
	orgID := defaultOrgID
	if id, err := multitenancy.GetOrgID(ctx); err == nil {
		orgID = id
		ctx = multitenancy.WithOrgID(ctx, id)
	} else {
		ctx = multitenancy.WithOrgID(ctx, defaultOrgID)
	}

	// Get buffer size from stream config
	bufferSize := 100
	if params.StreamConfig != nil {
		bufferSize = params.StreamConfig.BufferSize
	}

	eventChan := make(chan interfaces.StreamEvent, bufferSize)

	go func() {
		defer close(eventChan)

		builder := newMessageHistoryBuilder(c.logger)
		inputItems := builder.buildResponseInputItems(ctx, prompt, params.Memory)

		// Prepend system message when provided
		if params.SystemMessage != "" {
			inputItems = append([]responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(params.SystemMessage, responses.EasyInputMessageRoleSystem),
			}, inputItems...)
		}

		// Ensure we always send at least the current prompt
		if len(inputItems) == 0 {
			inputItems = []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(prompt, responses.EasyInputMessageRoleUser),
			}
		}

		req := responses.ResponseNewParams{
			Model: shared.ResponsesModel(c.Model),
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam(inputItems),
			},
			User: param.NewOpt(orgID),
		}

		if params.LLMConfig != nil {
			req.Temperature = param.NewOpt(c.getTemperatureForModel(params.LLMConfig.Temperature))
			if params.LLMConfig.TopP > 0 && !isReasoningModel(c.Model) {
				req.TopP = param.NewOpt(params.LLMConfig.TopP)
			}

			if params.LLMConfig.Reasoning != "" {
				req.Reasoning = shared.ReasoningParam{
					Effort: shared.ReasoningEffort(params.LLMConfig.Reasoning),
				}
			}
		}

		if params.ResponseFormat != nil {
			jsonSchema := responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   params.ResponseFormat.Name,
				Schema: params.ResponseFormat.Schema,
				Type:   "json_schema",
			}

			req.Text.Format = responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &jsonSchema,
			}
		}

		c.logger.Debug(ctx, "Creating OpenAI Responses streaming request", map[string]interface{}{
			"model":       c.Model,
			"temperature": params.LLMConfig.Temperature,
			"top_p":       params.LLMConfig.TopP,
			"reasoning":   params.LLMConfig.Reasoning,
		})

		stream := c.ResponseService.NewStreaming(ctx, req)

		eventChan <- interfaces.StreamEvent{
			Type:      interfaces.StreamEventMessageStart,
			Timestamp: time.Now(),
			Metadata: map[string]interface{}{
				"model": c.Model,
			},
		}

		includeThinking := params.StreamConfig == nil || params.StreamConfig.IncludeThinking
		var usageMetadata map[string]interface{}

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "response.output_text.delta":
				delta := event.AsResponseOutputTextDelta()
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventContentDelta,
					Content:   delta.Delta,
					Timestamp: time.Now(),
					Metadata: map[string]interface{}{
						"output_index":  delta.OutputIndex,
						"content_index": delta.ContentIndex,
					},
				}
			case "response.output_text.done":
				done := event.AsResponseOutputTextDone()
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventContentComplete,
					Timestamp: time.Now(),
					Metadata: map[string]interface{}{
						"output_index":  done.OutputIndex,
						"content_index": done.ContentIndex,
					},
				}
			case "response.reasoning_text.delta":
				if includeThinking {
					reasoning := event.AsResponseReasoningTextDelta()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventThinking,
						Content:   reasoning.Delta,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"output_index":  reasoning.OutputIndex,
							"content_index": reasoning.ContentIndex,
						},
					}
				}
			case "response.reasoning_text.done":
				if includeThinking {
					reasoning := event.AsResponseReasoningTextDone()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventThinking,
						Content:   reasoning.Text,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"output_index":  reasoning.OutputIndex,
							"content_index": reasoning.ContentIndex,
							"final":         true,
						},
					}
				}
			case "response.reasoning_summary_text.delta":
				if includeThinking {
					summary := event.AsResponseReasoningSummaryTextDelta()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventThinking,
						Content:   summary.Delta,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"summary_index": summary.SummaryIndex,
							"output_index":  summary.OutputIndex,
						},
					}
				}
			case "response.reasoning_summary_text.done":
				if includeThinking {
					summary := event.AsResponseReasoningSummaryTextDone()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventThinking,
						Content:   summary.Text,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"summary_index": summary.SummaryIndex,
							"output_index":  summary.OutputIndex,
							"final":         true,
						},
					}
				}
			case "response.completed":
				completed := event.AsResponseCompleted()
				if completed.Response.Usage.TotalTokens > 0 {
					usageMetadata = map[string]interface{}{
						"usage": map[string]interface{}{
							"prompt_tokens":       completed.Response.Usage.InputTokens,
							"completion_tokens":   completed.Response.Usage.OutputTokens,
							"reasoning_tokens":    completed.Response.Usage.OutputTokensDetails.ReasoningTokens,
							"total_tokens":        completed.Response.Usage.TotalTokens,
							"cached_input_tokens": completed.Response.Usage.InputTokensDetails.CachedTokens,
						},
					}
				}
			case "error":
				errEvent := event.AsError()
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventError,
					Error:     fmt.Errorf("openai streaming error: %s", errEvent.Message),
					Timestamp: time.Now(),
					Metadata: map[string]interface{}{
						"code": errEvent.Code,
					},
				}
			}
		}

		if err := stream.Err(); err != nil {
			c.logger.Error(ctx, "OpenAI streaming error", map[string]interface{}{
				"error": err.Error(),
				"model": c.Model,
			})
			eventChan <- interfaces.StreamEvent{
				Type:      interfaces.StreamEventError,
				Error:     fmt.Errorf("openai streaming error: %w", err),
				Timestamp: time.Now(),
			}
			return
		}

		eventChan <- interfaces.StreamEvent{
			Type:      interfaces.StreamEventMessageStop,
			Timestamp: time.Now(),
			Metadata:  usageMetadata,
		}

		c.logger.Debug(ctx, "Successfully completed OpenAI streaming request", map[string]interface{}{
			"model": c.Model,
		})
	}()

	return eventChan, nil
}

// GenerateWithToolsStream implements interfaces.StreamingLLM.GenerateWithToolsStream with iterative tool calling
func (c *OpenAIClient) GenerateWithToolsStream(
	ctx context.Context,
	prompt string,
	tools []interfaces.Tool,
	options ...interfaces.GenerateOption,
) (<-chan interfaces.StreamEvent, error) {
	// Apply options
	params := &interfaces.GenerateOptions{
		LLMConfig: &interfaces.LLMConfig{
			Temperature: 0.7,
		},
	}

	for _, option := range options {
		option(params)
	}

	// Set default max iterations if not provided
	maxIterations := params.MaxIterations
	if maxIterations == 0 {
		maxIterations = 2
	}

	// Check for organization ID in context
	defaultOrgID := "default"
	orgID := defaultOrgID
	if id, err := multitenancy.GetOrgID(ctx); err == nil {
		orgID = id
		ctx = multitenancy.WithOrgID(ctx, id)
	} else {
		ctx = multitenancy.WithOrgID(ctx, defaultOrgID)
	}

	// Get buffer size from stream config
	bufferSize := 100
	if params.StreamConfig != nil {
		bufferSize = params.StreamConfig.BufferSize
	}

	// Create event channel
	eventChan := make(chan interfaces.StreamEvent, bufferSize)

	go func() {
		defer close(eventChan)

		builder := newMessageHistoryBuilder(c.logger)
		inputItems := builder.buildResponseInputItems(ctx, prompt, params.Memory)

		// Prepend system message when provided
		if params.SystemMessage != "" {
			inputItems = append([]responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(params.SystemMessage, responses.EasyInputMessageRoleSystem),
			}, inputItems...)
		}

		// Ensure we always send at least the current prompt
		if len(inputItems) == 0 {
			inputItems = []responses.ResponseInputItemUnionParam{
				responses.ResponseInputItemParamOfMessage(prompt, responses.EasyInputMessageRoleUser),
			}
		}

		// Convert tools to Responses format
		responseTools := make([]responses.ToolUnionParam, len(tools))
		for i, tool := range tools {
			schema := c.convertToOpenAISchema(tool.Parameters())
			responseTools[i] = responses.ToolParamOfFunction(tool.Name(), schema, false)
		}

		// Send initial message start event
		eventChan <- interfaces.StreamEvent{
			Type:      interfaces.StreamEventMessageStart,
			Timestamp: time.Now(),
			Metadata: map[string]interface{}{
				"model": c.Model,
				"tools": len(responseTools),
			},
		}

		includeThinking := params.StreamConfig == nil || params.StreamConfig.IncludeThinking

		for iteration := 0; iteration < maxIterations; iteration++ {
			req := responses.ResponseNewParams{
				Model: shared.ResponsesModel(c.Model),
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: responses.ResponseInputParam(inputItems),
				},
				Tools: responseTools,
				User:  param.NewOpt(orgID),
			}

			if params.LLMConfig != nil {
				req.Temperature = param.NewOpt(c.getTemperatureForModel(params.LLMConfig.Temperature))
				if params.LLMConfig.TopP > 0 && !isReasoningModel(c.Model) {
					req.TopP = param.NewOpt(params.LLMConfig.TopP)
				}
				if params.LLMConfig.Reasoning != "" || includeThinking {
					req.Reasoning = shared.ReasoningParam{}
					if params.LLMConfig.Reasoning != "" {
						req.Reasoning.Effort = shared.ReasoningEffort(params.LLMConfig.Reasoning)
					}
					// Request reasoning summary when thinking is desired so we get reasoning_text deltas.
					if includeThinking {
						req.Reasoning.Summary = shared.ReasoningSummaryAuto
					}
				}
			}

			if params.ResponseFormat != nil {
				jsonSchema := responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   params.ResponseFormat.Name,
					Schema: params.ResponseFormat.Schema,
					Type:   "json_schema",
				}
				req.Text.Format = responses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: &jsonSchema,
				}
			}

			c.logger.Debug(ctx, "Creating OpenAI Responses streaming request with tools", map[string]interface{}{
				"model":         c.Model,
				"tools":         len(responseTools),
				"iteration":     iteration + 1,
				"maxIterations": maxIterations,
				"input_count":   len(inputItems),
			})

			stream := c.ResponseService.NewStreaming(ctx, req)
			if stream.Err() != nil {
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventError,
					Error:     fmt.Errorf("openai streaming error: %w", stream.Err()),
					Timestamp: time.Now(),
				}
				return
			}

			// Track tool calls by output item ID
			type toolCallState struct {
				callID    string
				name      string
				arguments strings.Builder
			}
			toolCalls := map[string]*toolCallState{}

			var usageMetadata map[string]interface{}
			var receivedContent bool

			for stream.Next() {
				event := stream.Current()

				switch event.Type {
				case "response.output_text.delta":
					delta := event.AsResponseOutputTextDelta()
					receivedContent = true
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventContentDelta,
						Content:   delta.Delta,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"output_index":  delta.OutputIndex,
							"content_index": delta.ContentIndex,
							"iteration":     iteration + 1,
						},
					}
				case "response.output_text.done":
					done := event.AsResponseOutputTextDone()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventContentComplete,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"output_index":  done.OutputIndex,
							"content_index": done.ContentIndex,
							"iteration":     iteration + 1,
						},
					}
				case "response.reasoning_text.delta":
					if includeThinking {
						reasoning := event.AsResponseReasoningTextDelta()
						eventChan <- interfaces.StreamEvent{
							Type:      interfaces.StreamEventThinking,
							Content:   reasoning.Delta,
							Timestamp: time.Now(),
							Metadata: map[string]interface{}{
								"output_index":  reasoning.OutputIndex,
								"content_index": reasoning.ContentIndex,
								"iteration":     iteration + 1,
							},
						}
					}
				case "response.reasoning_text.done":
					if includeThinking {
						reasoning := event.AsResponseReasoningTextDone()
						eventChan <- interfaces.StreamEvent{
							Type:      interfaces.StreamEventThinking,
							Content:   reasoning.Text,
							Timestamp: time.Now(),
							Metadata: map[string]interface{}{
								"output_index":  reasoning.OutputIndex,
								"content_index": reasoning.ContentIndex,
								"iteration":     iteration + 1,
								"final":         true,
							},
						}
					}
				case "response.reasoning_summary_text.delta":
					if includeThinking {
						summary := event.AsResponseReasoningSummaryTextDelta()
						eventChan <- interfaces.StreamEvent{
							Type:      interfaces.StreamEventThinking,
							Content:   summary.Delta,
							Timestamp: time.Now(),
							Metadata: map[string]interface{}{
								"summary_index": summary.SummaryIndex,
								"output_index":  summary.OutputIndex,
								"iteration":     iteration + 1,
							},
						}
					}
				case "response.reasoning_summary_text.done":
					if includeThinking {
						summary := event.AsResponseReasoningSummaryTextDone()
						eventChan <- interfaces.StreamEvent{
							Type:      interfaces.StreamEventThinking,
							Content:   summary.Text,
							Timestamp: time.Now(),
							Metadata: map[string]interface{}{
								"summary_index": summary.SummaryIndex,
								"output_index":  summary.OutputIndex,
								"iteration":     iteration + 1,
								"final":         true,
							},
						}
					}
				case "response.output_item.added":
					added := event.AsResponseOutputItemAdded()
					if added.Item.Type == "function_call" {
						toolCalls[added.Item.ID] = &toolCallState{
							callID: added.Item.CallID,
							name:   added.Item.Name,
						}
					}
				case "response.function_call_arguments.delta":
					delta := event.AsResponseFunctionCallArgumentsDelta()
					if tc, ok := toolCalls[delta.ItemID]; ok {
						tc.arguments.WriteString(delta.Delta)
					}
				case "response.function_call_arguments.done":
					done := event.AsResponseFunctionCallArgumentsDone()
					if tc, ok := toolCalls[done.ItemID]; ok {
						tc.arguments.WriteString(done.Arguments)
						eventChan <- interfaces.StreamEvent{
							Type: interfaces.StreamEventToolUse,
							ToolCall: &interfaces.ToolCall{
								ID:        tc.callID,
								Name:      tc.name,
								Arguments: tc.arguments.String(),
							},
							Timestamp: time.Now(),
							Metadata: map[string]interface{}{
								"iteration": iteration + 1,
							},
						}
					}
				case "response.completed":
					completed := event.AsResponseCompleted()
					if completed.Response.Usage.TotalTokens > 0 {
						usageMetadata = map[string]interface{}{
							"usage": map[string]interface{}{
								"prompt_tokens":       completed.Response.Usage.InputTokens,
								"completion_tokens":   completed.Response.Usage.OutputTokens,
								"reasoning_tokens":    completed.Response.Usage.OutputTokensDetails.ReasoningTokens,
								"total_tokens":        completed.Response.Usage.TotalTokens,
								"cached_input_tokens": completed.Response.Usage.InputTokensDetails.CachedTokens,
							},
						}
					}
				case "error":
					errEvent := event.AsError()
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventError,
						Error:     fmt.Errorf("openai streaming error: %s", errEvent.Message),
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"code": errEvent.Code,
						},
					}
				}
			}

			if err := stream.Err(); err != nil {
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventError,
					Error:     fmt.Errorf("openai streaming error: %w", err),
					Timestamp: time.Now(),
				}
				return
			}

			// If no tool calls, we're done
			if len(toolCalls) == 0 {
				if receivedContent {
					eventChan <- interfaces.StreamEvent{
						Type:      interfaces.StreamEventContentComplete,
						Timestamp: time.Now(),
						Metadata: map[string]interface{}{
							"iteration": iteration + 1,
						},
					}
				}
				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventMessageStop,
					Timestamp: time.Now(),
					Metadata:  usageMetadata,
				}
				return
			}

			// Execute tools and append outputs for next iteration
			var newItems []responses.ResponseInputItemUnionParam
			for _, tc := range toolCalls {
				var selected interfaces.Tool
				for _, t := range tools {
					if t.Name() == tc.name {
						selected = t
						break
					}
				}

				if selected == nil {
					continue
				}

				result, err := selected.Execute(ctx, tc.arguments.String())
				if err != nil {
					result = fmt.Sprintf("Error executing tool: %v", err)
				}

				eventChan <- interfaces.StreamEvent{
					Type:      interfaces.StreamEventToolResult,
					Timestamp: time.Now(),
					ToolCall: &interfaces.ToolCall{
						ID:        tc.callID,
						Name:      tc.name,
						Arguments: tc.arguments.String(),
					},
					Metadata: map[string]interface{}{
						"iteration": iteration + 1,
						"result":    result,
					},
				}

				newItems = append(newItems,
					responses.ResponseInputItemParamOfFunctionCall(tc.arguments.String(), tc.callID, tc.name),
					responses.ResponseInputItemParamOfFunctionCallOutput(tc.callID, result),
				)

				if params.Memory != nil {
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role: "assistant",
						ToolCalls: []interfaces.ToolCall{{
							ID:        tc.callID,
							Name:      tc.name,
							Arguments: tc.arguments.String(),
						}},
					})
					_ = params.Memory.AddMessage(ctx, interfaces.Message{
						Role:       "tool",
						Content:    result,
						ToolCallID: tc.callID,
						Metadata: map[string]interface{}{
							"tool_name": tc.name,
						},
					})
				}
			}

			inputItems = append(inputItems, newItems...)
		}

		// Max iterations reached without completion
		eventChan <- interfaces.StreamEvent{
			Type:      interfaces.StreamEventMessageStop,
			Timestamp: time.Now(),
		}
	}()

	return eventChan, nil
}

// convertToOpenAISchema converts tool parameters to OpenAI function schema
func (c *OpenAIClient) convertToOpenAISchema(params map[string]interfaces.ParameterSpec) map[string]interface{} {
	properties := make(map[string]interface{})
	required := []string{}

	for name, param := range params {
		property := map[string]interface{}{
			"type":        param.Type,
			"description": param.Description,
		}

		if param.Default != nil {
			property["default"] = param.Default
		}

		if param.Items != nil {
			property["items"] = map[string]interface{}{
				"type": param.Items.Type,
			}
			if param.Items.Enum != nil {
				property["items"].(map[string]interface{})["enum"] = param.Items.Enum
			}
		}

		if param.Enum != nil {
			property["enum"] = param.Enum
		}

		properties[name] = property

		if param.Required {
			required = append(required, name)
		}
	}

	return map[string]interface{}{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
}
