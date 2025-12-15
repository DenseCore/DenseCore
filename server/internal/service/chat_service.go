package service

import (
	"context"
	"errors"

	"descore-server/internal/domain"
)

type ChatService struct {
	modelService domain.ModelService
}

func NewChatService(modelService domain.ModelService) *ChatService {
	return &ChatService{
		modelService: modelService,
	}
}

// GenerateStream processes a chat completion request and streams tokens via the output channel.
// Accepts context.Context for propagating cancellation to the C++ engine.
// Returns an error if the model is not loaded or if the request is invalid.
func (s *ChatService) GenerateStream(ctx context.Context, req domain.ChatCompletionRequest, outputChan chan domain.StreamEvent) error {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return errors.New("no model loaded")
	}

	// Validate request parameters
	if req.MaxTokens < 0 {
		return errors.New("max_tokens must be non-negative")
	}
	if req.MaxTokens > 32000 {
		return errors.New("max_tokens exceeds maximum limit (32000)")
	}

	prompt := s.extractPrompt(req.Messages)
	if prompt == "" {
		return errors.New("no user message found")
	}

	// Check if JSON mode is requested
	jsonMode := req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"

	if jsonMode {
		return engine.GenerateStreamWithFormat(ctx, prompt, req.MaxTokens, true, outputChan)
	}

	return engine.GenerateStream(ctx, prompt, req.MaxTokens, outputChan)
}

func (s *ChatService) GetEmbeddings(req domain.EmbeddingRequest) ([]float32, error) {
	engine := s.modelService.GetEngine()
	if engine == nil {
		return nil, errors.New("no model loaded")
	}

	// Handle both string and []string input
	var inputText string
	switch v := req.Input.(type) {
	case string:
		inputText = v
	case []interface{}:
		if len(v) > 0 {
			if str, ok := v[0].(string); ok {
				inputText = str
			}
		}
	}

	return engine.GetEmbeddings(inputText)
}

func (s *ChatService) extractPrompt(messages []domain.Message) string {
	if len(messages) == 0 {
		return ""
	}
	// Find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return messages[len(messages)-1].Content
}
