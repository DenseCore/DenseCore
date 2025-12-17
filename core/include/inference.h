#ifndef DENSECORE_INFERENCE_H
#define DENSECORE_INFERENCE_H

#include "model_types.h"
#include <set>
#include <vector>

#include "kv_cache.h"

struct BatchSpec {
  std::vector<int> tokens; // Flattened tokens [TotalTokens]
  std::vector<int> pos;    // Position of each token [TotalTokens]
  std::vector<int> seq_id; // Sequence ID for each token [TotalTokens]
  std::vector<BlockTable> block_tables; // [NumSeqs]
  std::vector<int> n_past; // Past tokens count for each sequence [NumSeqs]
  int num_seqs;
};

// Inference configuration options
struct InferenceConfig {
  // Memory efficiency options
  bool use_flash_attention = false;    // Use memory-efficient attention
  int flash_attention_block_size = 64; // Tile size for Flash Attention

  // Prefetch options
  bool enable_prefetch = true; // Enable KV cache prefetching
  int prefetch_lookahead = 1;  // Layers to prefetch ahead

  // Performance options
  int num_threads = 1; // Thread count for parallel ops

  // Singleton instance
  static InferenceConfig &Instance() {
    static InferenceConfig config;
    return config;
  }
};

struct ggml_tensor *BuildTransformerGraph(
    TransformerModel *model, PagedKVCache *cache, struct ggml_context *ctx_c,
    const BatchSpec &batch, bool embedding_mode = false,
    struct ggml_cgraph *gf = nullptr, struct ggml_tensor **out_embd = nullptr,
    struct ggml_tensor **out_pos = nullptr);

// Initialize pre-computed RoPE cos/sin table for optimized inference
void InitRoPETable(TransformerModel *model);

// Grammar constraint for structured output (e.g., JSON mode)
enum class JSONState {
  EXPECT_OBJECT_START, // Expecting '{'
  EXPECT_KEY_OR_END,   // Expecting '"' (key) or '}'
  IN_KEY,              // Inside a key string
  EXPECT_COLON,        // Expecting ':'
  EXPECT_VALUE, // Expecting value (string, number, bool, null, object, array)
  IN_STRING_VALUE,     // Inside a string value
  EXPECT_COMMA_OR_END, // Expecting ',' or '}'
  IN_NUMBER,           // Inside a number value
  IN_ARRAY,            // Inside an array
  COMPLETED            // JSON object completed
};

struct GrammarConstraint {
  bool enabled = false;
  bool is_json_mode = false;
  JSONState state = JSONState::EXPECT_OBJECT_START;
  int brace_depth = 0;     // Track nested objects
  int bracket_depth = 0;   // Track nested arrays
  bool in_escape = false;  // Track escape sequences in strings
  std::string accumulated; // Accumulated output for state tracking

  // Token ID mappings (to be filled during initialization)
  int token_lbrace = -1;   // '{'
  int token_rbrace = -1;   // '}'
  int token_lbracket = -1; // '['
  int token_rbracket = -1; // ']'
  int token_quote = -1;    // '"'
  int token_colon = -1;    // ':'
  int token_comma = -1;    // ','

  // Helper: Update state after sampling a token
  void UpdateState(const std::string &token_text);
};

// Initialize grammar constraint with token mappings
void InitGrammarConstraint(GrammarConstraint *grammar,
                           const std::vector<std::string> &vocab);

// Apply grammar mask to logits based on current state
void ApplyGrammarMask(float *logits, int n_vocab,
                      const GrammarConstraint *grammar,
                      const std::vector<std::string> &vocab);

// Sampling parameters
struct SamplingParams {
  float temperature = 1.0f;
  int top_k = 40;
  float top_p = 0.95f;
  float min_p = 0.05f;
  float repetition_penalty = 1.0f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;

  // Token history for penalties
  const std::vector<int> *token_history = nullptr;

  // Grammar constraint for structured output
  GrammarConstraint *grammar = nullptr;

  // Vocabulary for grammar masking
  const std::vector<std::string> *vocab = nullptr;
};

int SampleToken(struct ggml_tensor *logits, int idx,
                const SamplingParams &params = SamplingParams());

#endif // DENSECORE_INFERENCE_H
