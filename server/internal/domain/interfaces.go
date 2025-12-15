package domain

import "context"

// Engine defines the interface for the inference engine
type Engine interface {
	GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan StreamEvent) error
	GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan StreamEvent) error
	GetEmbeddings(prompt string) ([]float32, error)
	GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error)
	GetMetrics() map[string]interface{}
	GetDetailedMetrics() *DetailedMetrics
	Close()
	CancelRequest(reqID uintptr)
}

// ModelService defines the interface for managing models
type ModelService interface {
	LoadModel(mainModelPath, draftModelPath string, threads int) error
	UnloadModel() error
	GetCurrentModel() string
	GetEngine() Engine
}
