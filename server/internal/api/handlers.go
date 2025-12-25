package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"descore-server/internal/domain"
	"descore-server/internal/service"
)

type Handler struct {
	chatService  *service.ChatService
	modelService domain.ModelService
}

func NewHandler(chatService *service.ChatService, modelService domain.ModelService) *Handler {
	return &Handler{
		chatService:  chatService,
		modelService: modelService,
	}
}

// ChatCompletionHandler handles OpenAI-compatible chat completion requests.
// Supports both streaming and synchronous responses.
func (h *Handler) ChatCompletionHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	var req domain.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON in request body", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	// Validate messages
	if len(req.Messages) == 0 {
		sendError(w, "messages field is required and must not be empty", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	// Set defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = 100
	}
	if req.MaxTokens < 0 {
		sendError(w, "max_tokens must be non-negative", "invalid_request_error", "", http.StatusBadRequest)
		return
	}
	if req.MaxTokens > 32000 {
		sendError(w, "max_tokens exceeds maximum limit (32000)", "invalid_request_error", "", http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = "densecore-v1"
	}

	slog.Info("processing chat completion request",
		slog.String("model", req.Model),
		slog.Int("max_tokens", req.MaxTokens),
		slog.Bool("stream", req.Stream),
	)

	// Extract context for cancellation propagation
	ctx := r.Context()

	if req.Stream {
		h.handleStream(ctx, w, req)
	} else {
		h.handleSync(ctx, w, req)
	}
}

func (h *Handler) handleStream(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	outputChan := make(chan domain.StreamEvent, 100)
	err := h.chatService.GenerateStream(ctx, req, outputChan)
	if err != nil {
		slog.Error("generation failed",
			slog.String("model", req.Model),
			slog.String("error", err.Error()),
		)
		close(outputChan)
		sendError(w, err.Error(), "internal_error", "", http.StatusInternalServerError)
		return
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().Unix())
	created := time.Now().Unix()

	for event := range outputChan {
		if event.IsFinished {
			if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
				slog.Debug("SSE write error", slog.String("error", err.Error()))
			}
			flusher.Flush()
			break
		}

		chunk := domain.ChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []domain.ChunkChoice{
				{
					Index: 0,
					Delta: domain.ChunkDelta{
						Content: event.Token,
					},
					FinishReason: nil,
				},
			},
		}

		data, _ := json.Marshal(chunk)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			slog.Debug("SSE write error", slog.String("error", err.Error()))
			break
		}
		flusher.Flush()
	}
}

func (h *Handler) handleSync(ctx context.Context, w http.ResponseWriter, req domain.ChatCompletionRequest) {
	outputChan := make(chan domain.StreamEvent, 100)
	var responseText string

	err := h.chatService.GenerateStream(ctx, req, outputChan)
	if err != nil {
		close(outputChan)
		sendError(w, err.Error(), "internal_error", "", http.StatusInternalServerError)
		return
	}

	for event := range outputChan {
		if !event.IsFinished {
			responseText += event.Token
		}
	}

	resp := domain.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []domain.Choice{
			{
				Index: 0,
				Message: domain.Message{
					Role:    "assistant",
					Content: responseText,
				},
				FinishReason: "stop",
			},
		},
		Usage: domain.Usage{
			CompletionTokens: len(responseText) / 4,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) EmbeddingsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	var req domain.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	texts := req.GetInputTexts()
	if len(texts) == 0 {
		sendError(w, "input is required", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	// Process each text and collect embeddings
	embeddings := make([]domain.EmbeddingData, 0, len(texts))
	totalTokens := 0

	for i, text := range texts {
		embd, err := h.chatService.GetEmbeddings(domain.EmbeddingRequest{Input: text})
		if err != nil {
			slog.Error("embedding generation failed",
				slog.Int("text_index", i),
				slog.String("error", err.Error()),
			)
			sendError(w, err.Error(), "internal_error", "", http.StatusInternalServerError)
			return
		}

		embeddings = append(embeddings, domain.EmbeddingData{
			Object:    "embedding",
			Embedding: embd,
			Index:     i,
		})
		totalTokens += len(text) / 4
	}

	resp := domain.EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model,
		Usage: domain.Usage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	currentModel := h.modelService.GetCurrentModel()

	response := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       "densecore-v1",
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "densecore",
				"root":     currentModel,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) LoadModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ModelPath      string `json:"model_path"`
		DraftModelPath string `json:"draft_model_path"`
		Threads        int    `json:"threads"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		sendError(w, "Invalid JSON", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	if req.ModelPath == "" {
		sendError(w, "model_path is required", "invalid_request_error", "", http.StatusBadRequest)
		return
	}

	if req.Threads == 0 {
		req.Threads = 4 // Default
	}

	err := h.modelService.LoadModel(req.ModelPath, req.DraftModelPath, req.Threads)
	if err != nil {
		sendError(w, fmt.Sprintf("Failed to load model: %v", err), "internal_error", "", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": fmt.Sprintf("Model loaded: %s", req.ModelPath),
	}); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) UnloadModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	err := h.modelService.UnloadModel()
	if err != nil {
		sendError(w, fmt.Sprintf("Failed to unload model: %v", err), "internal_error", "", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status":  "success",
		"message": "Model unloaded",
	}); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// HealthHandler - Basic health check (legacy)
func (h *Handler) HealthHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":  "ok",
		"version": "1.1.0",
		"engine":  "densecore",
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// LivenessHandler - K8s liveness probe
// Returns 200 if the process is alive
func (h *Handler) LivenessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// ReadinessHandler - K8s readiness probe
// Returns 200 only if the model is loaded and ready to serve
func (h *Handler) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	engine := h.modelService.GetEngine()
	if engine == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "not_ready",
			"reason": "model_not_loaded",
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	// Check if engine is healthy by getting metrics
	metrics := engine.GetDetailedMetrics()

	response := map[string]interface{}{
		"status":       "ready",
		"model":        h.modelService.GetCurrentModel(),
		"kv_cache_pct": metrics.KVCacheUsagePercent,
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
	}

	// Mark as not ready if KV cache is critically full
	if metrics.KVCacheUsagePercent > 95 {
		w.WriteHeader(http.StatusServiceUnavailable)
		response["status"] = "degraded"
		response["reason"] = "kv_cache_full"
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// StartupHandler - K8s startup probe
// Returns 200 once the initial model loading is complete
func (h *Handler) StartupHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	engine := h.modelService.GetEngine()
	if engine == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "starting",
			"progress": 0,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "started",
		"progress": 100,
		"model":    h.modelService.GetCurrentModel(),
	}); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// MetricsHandler outputs Prometheus-format metrics.
// Error checking for fmt.Fprintf is intentionally omitted per Prometheus exposition pattern.
//
//nolint:errcheck // Prometheus exposition format - write errors are non-recoverable
func (h *Handler) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", "", http.StatusMethodNotAllowed)
		return
	}

	engine := h.modelService.GetEngine()
	if engine == nil {
		http.Error(w, "No model loaded", http.StatusServiceUnavailable)
		return
	}

	// Get detailed metrics
	metrics := engine.GetDetailedMetrics()

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")

	// Request metrics
	fmt.Fprintf(w, "# HELP densecore_active_requests Number of currently active requests\n")
	fmt.Fprintf(w, "# TYPE densecore_active_requests gauge\n")
	fmt.Fprintf(w, "densecore_active_requests %d\n\n", metrics.ActiveRequests)

	fmt.Fprintf(w, "# HELP densecore_pending_requests Number of pending requests in queue\n")
	fmt.Fprintf(w, "# TYPE densecore_pending_requests gauge\n")
	fmt.Fprintf(w, "densecore_pending_requests %d\n\n", metrics.PendingRequests)

	fmt.Fprintf(w, "# HELP densecore_total_requests Total number of requests received\n")
	fmt.Fprintf(w, "# TYPE densecore_total_requests counter\n")
	fmt.Fprintf(w, "densecore_total_requests %d\n\n", metrics.TotalRequests)

	fmt.Fprintf(w, "# HELP densecore_completed_requests Total number of completed requests\n")
	fmt.Fprintf(w, "# TYPE densecore_completed_requests counter\n")
	fmt.Fprintf(w, "densecore_completed_requests %d\n\n", metrics.CompletedRequests)

	fmt.Fprintf(w, "# HELP densecore_failed_requests Total number of failed requests\n")
	fmt.Fprintf(w, "# TYPE densecore_failed_requests counter\n")
	fmt.Fprintf(w, "densecore_failed_requests %d\n\n", metrics.FailedRequests)

	// Token metrics
	fmt.Fprintf(w, "# HELP densecore_total_tokens_generated Total number of tokens generated\n")
	fmt.Fprintf(w, "# TYPE densecore_total_tokens_generated counter\n")
	fmt.Fprintf(w, "densecore_total_tokens_generated %d\n\n", metrics.TotalTokensGenerated)

	fmt.Fprintf(w, "# HELP densecore_total_prompt_tokens Total number of prompt tokens processed\n")
	fmt.Fprintf(w, "# TYPE densecore_total_prompt_tokens counter\n")
	fmt.Fprintf(w, "densecore_total_prompt_tokens %d\n\n", metrics.TotalPromptTokens)

	fmt.Fprintf(w, "# HELP densecore_tokens_per_second Average tokens generated per second\n")
	fmt.Fprintf(w, "# TYPE densecore_tokens_per_second gauge\n")
	fmt.Fprintf(w, "densecore_tokens_per_second %.2f\n\n", metrics.TokensPerSecond)

	// TTFT metrics
	fmt.Fprintf(w, "# HELP densecore_time_to_first_token_seconds Time to first token latency\n")
	fmt.Fprintf(w, "# TYPE densecore_time_to_first_token_seconds gauge\n")
	fmt.Fprintf(w, "densecore_time_to_first_token_seconds{quantile=\"0.5\"} %.6f\n", metrics.P50TimeToFirstToken/1000.0)
	fmt.Fprintf(w, "densecore_time_to_first_token_seconds{quantile=\"0.9\"} %.6f\n", metrics.P90TimeToFirstToken/1000.0)
	fmt.Fprintf(w, "densecore_time_to_first_token_seconds{quantile=\"0.99\"} %.6f\n", metrics.P99TimeToFirstToken/1000.0)
	fmt.Fprintf(w, "densecore_time_to_first_token_seconds{quantile=\"avg\"} %.6f\n\n", metrics.AvgTimeToFirstToken/1000.0)

	// ITL metrics
	fmt.Fprintf(w, "# HELP densecore_inter_token_latency_seconds Inter-token latency\n")
	fmt.Fprintf(w, "# TYPE densecore_inter_token_latency_seconds gauge\n")
	fmt.Fprintf(w, "densecore_inter_token_latency_seconds{quantile=\"0.5\"} %.6f\n", metrics.P50InterTokenLatency/1000.0)
	fmt.Fprintf(w, "densecore_inter_token_latency_seconds{quantile=\"0.9\"} %.6f\n", metrics.P90InterTokenLatency/1000.0)
	fmt.Fprintf(w, "densecore_inter_token_latency_seconds{quantile=\"0.99\"} %.6f\n", metrics.P99InterTokenLatency/1000.0)
	fmt.Fprintf(w, "densecore_inter_token_latency_seconds{quantile=\"avg\"} %.6f\n\n", metrics.AvgInterTokenLatency/1000.0)

	// Queue wait time
	fmt.Fprintf(w, "# HELP densecore_queue_wait_time_seconds Request queue wait time\n")
	fmt.Fprintf(w, "# TYPE densecore_queue_wait_time_seconds gauge\n")
	fmt.Fprintf(w, "densecore_queue_wait_time_seconds{quantile=\"0.99\"} %.6f\n", metrics.P99QueueWaitTime/1000.0)
	fmt.Fprintf(w, "densecore_queue_wait_time_seconds{quantile=\"avg\"} %.6f\n\n", metrics.AvgQueueWaitTime/1000.0)

	// KV Cache metrics
	fmt.Fprintf(w, "# HELP densecore_kv_cache_usage_blocks Number of KV cache blocks in use\n")
	fmt.Fprintf(w, "# TYPE densecore_kv_cache_usage_blocks gauge\n")
	fmt.Fprintf(w, "densecore_kv_cache_usage_blocks %d\n\n", metrics.KVCacheUsageBlocks)

	fmt.Fprintf(w, "# HELP densecore_kv_cache_total_blocks Total number of KV cache blocks\n")
	fmt.Fprintf(w, "# TYPE densecore_kv_cache_total_blocks gauge\n")
	fmt.Fprintf(w, "densecore_kv_cache_total_blocks %d\n\n", metrics.KVCacheTotalBlocks)

	fmt.Fprintf(w, "# HELP densecore_kv_cache_usage_percent KV cache usage percentage\n")
	fmt.Fprintf(w, "# TYPE densecore_kv_cache_usage_percent gauge\n")
	fmt.Fprintf(w, "densecore_kv_cache_usage_percent %.2f\n\n", metrics.KVCacheUsagePercent)

	// Batch metrics
	fmt.Fprintf(w, "# HELP densecore_current_batch_size Current batch size\n")
	fmt.Fprintf(w, "# TYPE densecore_current_batch_size gauge\n")
	fmt.Fprintf(w, "densecore_current_batch_size %d\n\n", metrics.CurrentBatchSize)

	fmt.Fprintf(w, "# HELP densecore_avg_batch_size Average batch size\n")
	fmt.Fprintf(w, "# TYPE densecore_avg_batch_size gauge\n")
	fmt.Fprintf(w, "densecore_avg_batch_size %.2f\n\n", metrics.AvgBatchSize)

	// Error metrics
	fmt.Fprintf(w, "# HELP densecore_oom_errors Total number of OOM errors\n")
	fmt.Fprintf(w, "# TYPE densecore_oom_errors counter\n")
	fmt.Fprintf(w, "densecore_oom_errors %d\n\n", metrics.OOMErrors)

	fmt.Fprintf(w, "# HELP densecore_timeout_errors Total number of timeout errors\n")
	fmt.Fprintf(w, "# TYPE densecore_timeout_errors counter\n")
	fmt.Fprintf(w, "densecore_timeout_errors %d\n", metrics.TimeoutErrors)
}

func sendError(w http.ResponseWriter, message, errType, code string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := domain.ErrorResponse{
		Error: domain.ErrorDetail{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	}

	if err := json.NewEncoder(w).Encode(errorResp); err != nil {
		slog.Debug("failed to encode error response", slog.String("error", err.Error()))
	}
}
