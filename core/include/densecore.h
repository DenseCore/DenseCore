/**
 * @file densecore.h
 * @brief Main DenseCore C API for LLM inference
 *
 * DenseCore provides high-performance CPU-based inference for large language
 * models with support for streaming generation, embeddings, and multi-model
 * management.
 *
 * Key features:
 * - Streaming token generation with callbacks
 * - Text embeddings with configurable pooling
 * - Paged KV cache for memory efficiency
 * - Multi-model management
 * - Comprehensive metrics and monitoring
 *
 * @section example_usage Example Usage
 * @code
 * // Initialize engine
 * DenseCoreHandle engine = InitEngine("model.gguf", NULL, 4);
 *
 * // Generate text with streaming
 * void callback(const char *token, int is_finished, void *user_data) {
 *     printf("%s", token);
 *     if (is_finished) printf("\n");
 * }
 *
 * SubmitRequest(engine, "Hello, how are you?", 100, callback, NULL);
 *
 * // Cleanup
 * FreeEngine(engine);
 * @endcode
 *
 * @author DenseCore Team
 * @version 1.1.0
 */

#ifndef DENSECORE_H
#define DENSECORE_H

// =============================================================================
// Symbol Visibility Macros
// =============================================================================
// DENSECORE_API marks functions for export in the shared library.
// All public API functions must be marked with this macro.
// Internal functions remain hidden (default visibility) for smaller binaries.
// =============================================================================
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef DENSECORE_BUILD_SHARED
#define DENSECORE_API __declspec(dllexport)
#else
#define DENSECORE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define DENSECORE_API __attribute__((visibility("default")))
#else
#define DENSECORE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version Information API
// =============================================================================

/**
 * @brief Version information structure
 */
typedef struct {
  int major;              ///< Major version number
  int minor;              ///< Minor version number
  int patch;              ///< Patch version number
  const char *version;    ///< Version string "X.Y.Z"
  const char *commit;     ///< Git commit hash (short)
  const char *build_time; ///< Build timestamp (ISO 8601)
  const char *full;       ///< Full version "X.Y.Z (commit)"
} DenseCoreVersionInfo;

/**
 * @brief Get the library version information
 *
 * Returns compile-time version information for the DenseCore library.
 * Useful for debugging and ensuring SDK/library version compatibility.
 *
 * @return Pointer to static VersionInfo structure (never NULL)
 *
 * @code
 * const DenseCoreVersionInfo *ver = GetLibraryVersion();
 * printf("DenseCore %s (commit: %s)\n", ver->version, ver->commit);
 * @endcode
 */
DENSECORE_API const DenseCoreVersionInfo *GetLibraryVersion(void);

/**
 * @brief Get the library version as a simple string
 *
 * @return Version string "X.Y.Z" (static, never NULL)
 */
DENSECORE_API const char *GetLibraryVersionString(void);

/// Opaque handle to the DenseCore engine
typedef void *DenseCoreHandle;

/**
 * @brief Structured token result for callbacks
 *
 * This structure provides both the token ID (for HuggingFace tokenizer
 * decoding) and the pre-decoded text. When using HF tokenizers in Python, use
 * token_id for accurate decoding.
 */
typedef struct {
  int token_id; ///< Token ID for external tokenizer decoding
  const char
      *text; ///< Pre-decoded text (may be empty if using external tokenizer)
  int is_finished; ///< 1 if generation is complete, 0 otherwise
} TokenResult;

/**
 * @brief Callback function for streaming tokens during generation
 *
 * This callback is invoked for each generated token during text generation.
 * The callback should be thread-safe if used in multi-threaded contexts.
 *
 * @param token The generated token as a UTF-8 string (null-terminated)
 * @param is_finished 1 if this is the final token/chunk, 0 otherwise
 * @param user_data User-provided pointer passed back to the callback
 *
 * @note The token pointer is only valid during the callback execution.
 *       Copy the string if you need to retain it.
 */
typedef void (*TokenCallback)(const char *token, int is_finished,
                              void *user_data);

/**
 * @brief Callback function for structured token results
 *
 * Enhanced callback that provides token ID for HuggingFace tokenizer decoding.
 *
 * @param result Pointer to TokenResult structure
 * @param user_data User-provided pointer
 */
typedef void (*TokenResultCallback)(const TokenResult *result, void *user_data);

/**
 * @brief Callback function for returning embeddings
 *
 * @param embedding Pointer to the embedding vector (array of floats)
 * @param size Dimension of the embedding vector
 * @param user_data User-provided pointer passed back to the callback
 *
 * @note The embedding pointer is only valid during the callback execution.
 *       Copy the data if you need to retain it.
 */
typedef void (*EmbeddingCallback)(const float *embedding, int size,
                                  void *user_data);

/**
 * @brief Initialize the DenseCore inference engine
 *
 * Loads a GGUF model and initializes the inference engine with the specified
 * configuration. This is a blocking operation that may take several seconds
 * for large models.
 *
 * @param model_path Path to the GGUF model file (required)
 * @param reserved Reserved for future use (pass NULL)
 * @param threads Number of CPU threads to use (0 for auto-detect)
 * @param numa_node_id NUMA node to bind memory and threads (-1 for
 * auto/default)
 * @param pinning_policy Thread pinning policy for compute threads:
 *        - 0 = SCATTER (default): Distribute threads across physical cores,
 *          maximizes L3 cache and memory bandwidth. Best for latency-sensitive
 *          single-user workloads.
 *        - 1 = COMPACT: Pack threads on adjacent cores, shares L2 cache.
 *          Best for throughput-oriented batch processing, leaves cores for
 *          other processes.
 * @return Handle to the initialized engine, or NULL on failure
 *
 * @note The engine must be freed with FreeEngine() when no longer needed.
 * @note For multi-socket servers, specifying numa_node_id can significantly
 *       improve performance by reducing cross-socket memory access.
 *
 * @see FreeEngine()
 *
 * Example:
 * @code
 * // Auto-detect NUMA with SCATTER pinning (default)
 * DenseCoreHandle engine = InitEngine("model.gguf", NULL, 4, -1, 0);
 *
 * // NUMA node 0 with COMPACT pinning for batch throughput
 * DenseCoreHandle engine = InitEngine("model.gguf", NULL, 4, 0, 1);
 * @endcode
 */
DENSECORE_API DenseCoreHandle InitEngine(const char *model_path,
                                         const char *reserved, int threads);

/**
 * Submit a request to the DenseCore engine (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequest(DenseCoreHandle handle, const char *prompt,
                                int max_tokens, TokenCallback callback,
                                void *user_data);

/**
 * Submit a request with pre-tokenized input (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens
 * @param max_tokens Maximum number of tokens to generate
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestIds(DenseCoreHandle handle, const int *tokens,
                                   int n_tokens, int max_tokens,
                                   TokenCallback callback, void *user_data);

/**
 * Submit a request with response format specification (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum number of tokens to generate
 * @param json_mode Enable JSON output mode (1 for JSON, 0 for text)
 * @param callback Function pointer for streaming tokens
 * @param user_data User data to pass to the callback
 * @return Request ID (positive integer) on success, or negative error code
 */
DENSECORE_API int SubmitRequestWithFormat(DenseCoreHandle handle,
                                          const char *prompt, int max_tokens,
                                          int json_mode, TokenCallback callback,
                                          void *user_data);

/**
 * Submit an embedding request (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input text
 * @param callback Function pointer for returning embedding
 * @param user_data User data
 * @return Request ID
 */
DENSECORE_API int SubmitEmbeddingRequest(DenseCoreHandle handle,
                                         const char *prompt,
                                         EmbeddingCallback callback,
                                         void *user_data);

/**
 * Submit an embedding request with options (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompt Input text
 * @param pooling_type Pooling strategy: 0=MEAN, 1=CLS, 2=LAST, 3=MAX
 * @param normalize Whether to L2 normalize (1=yes, 0=no)
 * @param callback Function pointer for returning embedding
 * @param user_data User data
 * @return Request ID
 */
DENSECORE_API int SubmitEmbeddingRequestEx(DenseCoreHandle handle,
                                           const char *prompt, int pooling_type,
                                           int normalize,
                                           EmbeddingCallback callback,
                                           void *user_data);

/**
 * Submit a batch embedding request (Non-blocking)
 *
 * @param handle Handle to the DenseCore engine
 * @param prompts Array of input texts
 * @param num_prompts Number of prompts
 * @param pooling_type Pooling strategy: 0=MEAN, 1=CLS, 2=LAST, 3=MAX
 * @param normalize Whether to L2 normalize
 * @param callback Callback for each embedding (called num_prompts times)
 * @param user_data User data
 * @return Request ID for the batch
 */
DENSECORE_API int
SubmitBatchEmbeddingRequest(DenseCoreHandle handle, const char **prompts,
                            int num_prompts, int pooling_type, int normalize,
                            EmbeddingCallback callback, void *user_data);

/**
 * Get the embedding dimension of the loaded model
 *
 * @param handle Handle to the DenseCore engine
 * @return Embedding dimension, or -1 on error
 */
DENSECORE_API int GetEmbeddingDimension(DenseCoreHandle handle);

/**
 * Cancel a running request
 *
 * @param handle Handle to the DenseCore engine
 * @param request_id ID of the request to cancel
 * @return 0 on success, non-zero on failure
 */
DENSECORE_API int CancelRequest(DenseCoreHandle handle, int request_id);

/**
 * Free the DenseCore engine and release resources
 *
 * @param handle Handle to the DenseCore engine
 */
DENSECORE_API void FreeEngine(DenseCoreHandle handle);

// Metrics API
typedef struct {
  float requests_per_second;
  float tokens_per_second;
  int active_requests;
  long total_tokens_generated;
} DenseCoreMetrics;

// Detailed Metrics API
typedef struct {
  // Request metrics
  int active_requests;
  long total_requests;
  long completed_requests;
  long failed_requests;
  int pending_requests;

  // Token metrics
  long total_tokens_generated;
  long total_prompt_tokens;
  float tokens_per_second;

  // Latency metrics (milliseconds)
  float avg_time_to_first_token;
  float p50_time_to_first_token;
  float p90_time_to_first_token;
  float p99_time_to_first_token;

  float avg_inter_token_latency;
  float p50_inter_token_latency;
  float p90_inter_token_latency;
  float p99_inter_token_latency;

  float avg_queue_wait_time;
  float p99_queue_wait_time;

  // KV Cache metrics
  int kv_cache_usage_blocks;
  int kv_cache_total_blocks;
  float kv_cache_usage_percent;

  // Batch metrics
  float avg_batch_size;
  int current_batch_size;

  // Error metrics
  int oom_errors;
  int timeout_errors;
} DetailedMetrics;

/**
 * Get current metrics
 * @param handle Handle to the DenseCore engine
 * @return Metrics structure
 */
DENSECORE_API DenseCoreMetrics GetMetrics(DenseCoreHandle handle);

/**
 * Get detailed metrics with latency percentiles
 * @param handle Handle to the DenseCore engine
 * @return Detailed metrics structure
 */
DENSECORE_API DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle);

// Multi-Model API

/**
 * Load a new model into the engine pool
 * @param handle Handle to the DenseCore engine
 * @param model_id Unique identifier for this model
 * @param model_path Path to the model file
 * @param threads Number of threads (0 for default)
 * @return 0 on success, negative on failure
 */
DENSECORE_API int LoadModel(DenseCoreHandle handle, const char *model_id,
                            const char *model_path, int threads);

/**
 * Unload a model from the engine pool
 * @param handle Handle to the DenseCore engine
 * @param model_id Model identifier to unload
 * @return 0 on success, negative on failure
 */
DENSECORE_API int UnloadModel(DenseCoreHandle handle, const char *model_id);

/**
 * List all loaded models
 * @param handle Handle to the DenseCore engine
 * @param out_models Output buffer for model IDs (comma-separated)
 * @param buffer_size Size of output buffer
 * @return Number of models, or negative on error
 */
DENSECORE_API int ListModels(DenseCoreHandle handle, char *out_models,
                             int buffer_size);

/**
 * Set the default model for requests
 * @param handle Handle to the DenseCore engine
 * @param model_id Model identifier to set as default
 * @return 0 on success, negative on failure
 */
DENSECORE_API int SetDefaultModel(DenseCoreHandle handle, const char *model_id);

/**
 * Quantize a model (Offline)
 * @param model_path Path to input GGUF model
 * @param output_path Path to save quantized model
 * @param config_json JSON string with quantization config (format, algo, etc.)
 * @return 0 on success, negative on error
 */
DENSECORE_API int QuantizeModel(const char *model_path, const char *output_path,
                                const char *config_json);

/**
 * Prune a model (Offline)
 * @param model_path Path to input GGUF model
 * @param output_path Path to save pruned model
 * @param config_json JSON string with pruning config (target layers/dim, etc.)
 * @return 0 on success, negative on error
 */
DENSECORE_API int PruneModel(const char *model_path, const char *output_path,
                             const char *config_json);

// =============================================================================
// LoRA Adapter Runtime API
// =============================================================================

/**
 * Load a LoRA adapter into the engine
 *
 * @param handle Handle to the DenseCore engine
 * @param path Path to the GGUF LoRA adapter file
 * @param scale LoRA scaling factor (alpha). 1.0 = full adapter effect
 * @param name Unique identifier for this adapter
 * @return 0 on success, negative on error
 */
DENSECORE_API int LoadLoraAdapter(DenseCoreHandle handle, const char *path,
                                  float scale, const char *name);

/**
 * Activate a loaded LoRA adapter
 *
 * @param handle Handle to the DenseCore engine
 * @param name Identifier of the adapter to activate
 * @return 0 on success, negative on error
 */
DENSECORE_API int ActivateLoraAdapter(DenseCoreHandle handle, const char *name);

/**
 * Deactivate all LoRA adapters (use base model only)
 *
 * @param handle Handle to the DenseCore engine
 * @return 0 on success, negative on error
 */
DENSECORE_API int DeactivateLoraAdapters(DenseCoreHandle handle);

/**
 * Unload a LoRA adapter from the engine
 *
 * @param handle Handle to the DenseCore engine
 * @param name Identifier of the adapter to unload
 * @return 0 on success, negative on error
 */
DENSECORE_API int UnloadLoraAdapter(DenseCoreHandle handle, const char *name);

#ifdef __cplusplus
}
#endif

#endif // DENSECORE_H