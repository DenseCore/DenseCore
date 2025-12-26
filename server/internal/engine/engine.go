package engine

/*
// CGO Build Configuration
// -----------------------
// IMPORTANT: For CI/production, set environment variables:
//   export CGO_CFLAGS="-I${PWD}/core/include"
//   export CGO_LDFLAGS="-L${PWD}/build -ldensecore -lstdc++"
//
// ${SRCDIR} expands to the directory containing this Go source file.
#cgo CFLAGS: -I${SRCDIR}/../../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build -ldensecore -lstdc++
#include <stdlib.h>
#include "densecore.h"

// Forward declaration of the Go callback (exported from callbacks.go)
extern void streamCallbackGateway(char* token, int is_finished, void* user_data);

// Wrapper function to call SubmitRequest with the callback
static int SubmitRequestWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, void* user_data) {
    return SubmitRequest(handle, prompt, max_tokens, (TokenCallback)streamCallbackGateway, user_data);
}

// Wrapper function for SubmitRequestWithFormat
static int SubmitRequestWithFormatWrapper(DenseCoreHandle handle, const char* prompt, int max_tokens, int json_mode, void* user_data) {
    return SubmitRequestWithFormat(handle, prompt, max_tokens, json_mode, (TokenCallback)streamCallbackGateway, user_data);
}

// Forward declaration of the Go callback for embeddings (exported from callbacks.go)
extern void embeddingCallbackGateway(float* embedding, int size, void* user_data);

// Wrapper for SubmitEmbeddingRequest
static int SubmitEmbeddingRequestWrapper(DenseCoreHandle handle, const char* prompt, void* user_data) {
    return SubmitEmbeddingRequest(handle, prompt, (EmbeddingCallback)embeddingCallbackGateway, user_data);
}

// Wrapper for SubmitEmbeddingRequestEx with pooling options
static int SubmitEmbeddingRequestExWrapper(DenseCoreHandle handle, const char* prompt, int pooling_type, int normalize, void* user_data) {
    return SubmitEmbeddingRequestEx(handle, prompt, pooling_type, normalize, (EmbeddingCallback)embeddingCallbackGateway, user_data);
}

// Wrapper for CancelRequest
static void CancelRequestWrapper(DenseCoreHandle handle, int request_id) {
    CancelRequest(handle, request_id);
}
*/
import "C"

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"descore-server/internal/domain"
)

var cleanupOnce sync.Once

// DenseEngine wraps the C++ engine with thread safety
type DenseEngine struct {
	handle C.DenseCoreHandle
	mu     sync.Mutex
}

// Counter for request IDs to use as user_data
var requestIDCounter uint64

// Initialize the DenseCore engine
func NewDenseEngine(mainModelPath, draftModelPath string, threads int) (*DenseEngine, error) {
	cMainPath := C.CString(mainModelPath)
	defer C.free(unsafe.Pointer(cMainPath))

	var cDraftPath *C.char
	if draftModelPath != "" {
		cDraftPath = C.CString(draftModelPath)
		defer C.free(unsafe.Pointer(cDraftPath))
	}

	// InitEngine(model_path, reserved, threads, numa_node_id, pinning_policy)
	// numa_node_id=-1 (auto), pinning_policy=0 (SCATTER - default)
	handle := C.InitEngine(cMainPath, cDraftPath, C.int(threads), C.int(-1), C.int(0))
	if handle == nil {
		return nil, fmt.Errorf("failed to initialize DenseCore engine")
	}

	// Ensure background cleanup ticker runs exactly once (global state)
	cleanupOnce.Do(func() {
		// Interval: 1 minute, TTL: 5 minutes
		StartMapCleanupTicker(1*time.Minute, 5*time.Minute)
	})

	return &DenseEngine{
		handle: handle,
	}, nil
}

// Generate response using the engine with streaming support (Non-blocking)
// Accepts context.Context for cancellation propagation to C++ engine.
func (e *DenseEngine) GenerateStream(ctx context.Context, prompt string, maxTokens int, outputChan chan domain.StreamEvent) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	// Call C wrapper
	//nolint:govet // G103: unsafe.Pointer is required for CGO callback user_data
	ret := C.SubmitRequestWrapper(e.handle, cPrompt, C.int(maxTokens), unsafe.Pointer(reqID))
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// GenerateStreamWithFormat generates response with specified output format (JSON mode)
// Accepts context.Context for cancellation propagation to C++ engine.
func (e *DenseEngine) GenerateStreamWithFormat(ctx context.Context, prompt string, maxTokens int, jsonMode bool, outputChan chan domain.StreamEvent) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	// Register request channel
	id := atomic.AddUint64(&requestIDCounter, 1)
	reqID := uintptr(id)
	requestChannels.Store(reqID, outputChan)

	// Register completion channel BEFORE submitting to avoid race
	completionCh := completionChannels.Register(reqID)

	// Convert bool to int for C
	jsonModeInt := 0
	if jsonMode {
		jsonModeInt = 1
	}

	// Call C wrapper with format
	//nolint:govet // G103: unsafe.Pointer is required for CGO callback user_data
	ret := C.SubmitRequestWithFormatWrapper(e.handle, cPrompt, C.int(maxTokens), C.int(jsonModeInt), unsafe.Pointer(reqID))
	if ret < 0 {
		Cleanup(reqID)
		return fmt.Errorf("submission failed with error code %d", ret)
	}

	// Launch context watcher goroutine
	go e.watchContext(ctx, reqID, completionCh)

	return nil
}

// watchContext monitors for context cancellation and cancels the C++ request if needed.
// This goroutine exits when either:
// 1. The context is canceled (and we call CancelRequest), or
// 2. The request completes normally (completionCh is closed)
func (e *DenseEngine) watchContext(ctx context.Context, reqID uintptr, completionCh <-chan struct{}) {
	select {
	case <-ctx.Done():
		// Context canceled - signal C++ to stop generating
		log.Printf("Context canceled for request %d, cleaning up...", reqID)
		e.CancelRequest(reqID) // C++ side cancellation
		Cleanup(reqID)         // Go side cleanup
	case <-completionCh:
		// Generation finished normally - exit silently
	}
}

// CancelRequest signals the C++ engine to cancel a running request.
// Safe to call even if the request has already completed.
func (e *DenseEngine) CancelRequest(reqID uintptr) {
	C.CancelRequestWrapper(e.handle, C.int(reqID))
}

// GetEmbeddings (Non-blocking) - default: MEAN pooling with normalization
func (e *DenseEngine) GetEmbeddings(prompt string) ([]float32, error) {
	return e.GetEmbeddingsWithOptions(prompt, "mean", nil)
}

// GetEmbeddingsWithOptions - with pooling type and normalization control
func (e *DenseEngine) GetEmbeddingsWithOptions(prompt string, poolingType string, normalize *bool) ([]float32, error) {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	outputChan := make(chan []float32, 1)
	id := atomic.AddUint64(&requestIDCounter, 1)
	embeddingChannels.Store(uintptr(id), outputChan)

	// Map pooling type to integer
	poolingInt := 0 // default: MEAN
	switch poolingType {
	case "mean":
		poolingInt = 0
	case "cls":
		poolingInt = 1
	case "last":
		poolingInt = 2
	case "max":
		poolingInt = 3
	}

	// Default normalize to true
	normalizeInt := 1
	if normalize != nil && !*normalize {
		normalizeInt = 0
	}

	//nolint:govet // G103: unsafe.Pointer is required for CGO callback user_data
	ret := C.SubmitEmbeddingRequestExWrapper(e.handle, cPrompt, C.int(poolingInt), C.int(normalizeInt), unsafe.Pointer(uintptr(id)))
	if ret < 0 {
		embeddingChannels.Delete(uintptr(id))
		return nil, fmt.Errorf("submission failed with error code %d", ret)
	}

	// Wait for result
	select {
	case embd := <-outputChan:
		return embd, nil
	case <-time.After(30 * time.Second): // Timeout
		embeddingChannels.Delete(uintptr(id))
		return nil, fmt.Errorf("timeout waiting for embeddings")
	}
}

// Free the engine
func (e *DenseEngine) Close() {
	if e.handle != nil {
		C.FreeEngine(e.handle)
		e.handle = nil
	}
}

// Get metrics from the engine
func (e *DenseEngine) GetMetrics() map[string]interface{} {
	cMetrics := C.GetMetrics(e.handle)

	return map[string]interface{}{
		"requests_per_second":    float32(cMetrics.requests_per_second),
		"tokens_per_second":      float32(cMetrics.tokens_per_second),
		"active_requests":        int(cMetrics.active_requests),
		"total_tokens_generated": int64(cMetrics.total_tokens_generated),
	}
}

// GetDetailedMetrics returns detailed metrics with latency percentiles
func (e *DenseEngine) GetDetailedMetrics() *domain.DetailedMetrics {
	cMetrics := C.GetDetailedMetrics(e.handle)

	return &domain.DetailedMetrics{
		ActiveRequests:    int(cMetrics.active_requests),
		TotalRequests:     int64(cMetrics.total_requests),
		CompletedRequests: int64(cMetrics.completed_requests),
		FailedRequests:    int64(cMetrics.failed_requests),
		PendingRequests:   int(cMetrics.pending_requests),

		TotalTokensGenerated: int64(cMetrics.total_tokens_generated),
		TotalPromptTokens:    int64(cMetrics.total_prompt_tokens),
		TokensPerSecond:      float32(cMetrics.tokens_per_second),

		AvgTimeToFirstToken: float32(cMetrics.avg_time_to_first_token),
		P50TimeToFirstToken: float32(cMetrics.p50_time_to_first_token),
		P90TimeToFirstToken: float32(cMetrics.p90_time_to_first_token),
		P99TimeToFirstToken: float32(cMetrics.p99_time_to_first_token),

		AvgInterTokenLatency: float32(cMetrics.avg_inter_token_latency),
		P50InterTokenLatency: float32(cMetrics.p50_inter_token_latency),
		P90InterTokenLatency: float32(cMetrics.p90_inter_token_latency),
		P99InterTokenLatency: float32(cMetrics.p99_inter_token_latency),

		AvgQueueWaitTime: float32(cMetrics.avg_queue_wait_time),
		P99QueueWaitTime: float32(cMetrics.p99_queue_wait_time),

		KVCacheUsageBlocks:  int(cMetrics.kv_cache_usage_blocks),
		KVCacheTotalBlocks:  int(cMetrics.kv_cache_total_blocks),
		KVCacheUsagePercent: float32(cMetrics.kv_cache_usage_percent),

		AvgBatchSize:     float32(cMetrics.avg_batch_size),
		CurrentBatchSize: int(cMetrics.current_batch_size),

		OOMErrors:     int(cMetrics.oom_errors),
		TimeoutErrors: int(cMetrics.timeout_errors),
	}
}
