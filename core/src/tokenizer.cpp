#include "tokenizer.h"
#include <queue> // Required for std::priority_queue

/**
 * BPE Tokenizer Implementation - Optimized with Priority Queue
 *
 * This implements the standard Byte-Pair Encoding algorithm with O(N log N)
 * complexity using a priority queue instead of O(N²) linear search:
 *
 * 1. Split input into initial tokens (characters or bytes)
 * 2. Build priority queue of all valid mergeable pairs
 * 3. Pop highest-priority pair, merge, update adjacent pairs
 * 4. Continue until no more merges are possible
 *
 * The merge priority is determined by token_scores from the GGUF vocab.
 * Lower scores = higher priority (merged first).
 */

// ============================================================================
// UTF-8 Utilities
// ============================================================================

/**
 * Get the number of bytes in a UTF-8 character starting at the given byte.
 */
static int Utf8CharLen(unsigned char c) {
  if ((c & 0x80) == 0)
    return 1; // 0xxxxxxx - ASCII
  if ((c & 0xE0) == 0xC0)
    return 2; // 110xxxxx - 2-byte sequence
  if ((c & 0xF0) == 0xE0)
    return 3; // 1110xxxx - 3-byte sequence
  if ((c & 0xF8) == 0xF0)
    return 4; // 11110xxx - 4-byte sequence
  return 1;   // Invalid, treat as single byte
}

std::vector<std::string> Tokenizer::SplitToChars(const std::string &text) {
  std::vector<std::string> chars;
  chars.reserve(text.size()); // Approximate

  size_t i = 0;
  while (i < text.size()) {
    int len = Utf8CharLen(static_cast<unsigned char>(text[i]));

    // Clamp to remaining string length
    if (i + len > text.size()) {
      len = text.size() - i;
    }

    chars.push_back(text.substr(i, len));
    i += len;
  }

  return chars;
}

// ============================================================================
// BPE Core
// ============================================================================

bool Tokenizer::HasToken(const TransformerModel *model,
                         const std::string &token) {
  return model->token_to_id.find(token) != model->token_to_id.end();
}

float Tokenizer::GetTokenScore(const TransformerModel *model,
                               const std::string &token) {
  auto it = model->token_to_id.find(token);
  if (it == model->token_to_id.end()) {
    return std::numeric_limits<float>::max(); // Not in vocab
  }

  int id = it->second;

  // If we have scores, use them; otherwise use ID as a proxy
  // (lower ID = more common = higher priority for merge)
  if (!model->token_scores.empty() && id < (int)model->token_scores.size()) {
    return model->token_scores[id];
  }

  // Fallback: use negative ID so lower IDs have higher priority
  return static_cast<float>(id);
}

// Legacy O(N²) fallback - kept for reference but no longer used
int Tokenizer::FindBestMerge(const TransformerModel *model,
                             const std::vector<std::string> &tokens) {
  if (tokens.size() < 2) {
    return -1; // Nothing to merge
  }

  int best_idx = -1;
  float best_score = std::numeric_limits<float>::max();

  for (size_t i = 0; i < tokens.size() - 1; i++) {
    std::string merged = tokens[i] + tokens[i + 1];

    // Check if merged token exists in vocabulary
    if (HasToken(model, merged)) {
      float score = GetTokenScore(model, merged);

      // Lower score = higher priority
      if (score < best_score) {
        best_score = score;
        best_idx = static_cast<int>(i);
      }
    }
  }

  return best_idx;
}

// ============================================================================
// Priority Queue BPE Merge - O(N log N) Algorithm
// ============================================================================

namespace {

/**
 * Represents a mergeable pair in the priority queue.
 * Includes position and generation to handle invalidated entries.
 */
struct MergePair {
  float score;
  size_t position;   // Index in token list where merge would occur
  size_t generation; // Invalidation counter for lazy deletion

  // Max-heap by default, so we invert comparison (lower score = higher
  // priority)
  bool operator<(const MergePair &other) const {
    return score > other.score; // Invert for min-heap behavior
  }
};

} // namespace

/**
 * Optimized BPE merge using priority queue.
 * Time complexity: O(N log N) instead of O(N²)
 */
static std::vector<std::string>
MergeWithPriorityQueue(const TransformerModel *model,
                       std::vector<std::string> tokens) {
  if (tokens.size() < 2) {
    return tokens;
  }

  // Generation counter for lazy deletion (each position has its own counter)
  std::vector<size_t> generation(tokens.size(), 0);

  // Priority queue of valid merge candidates
  std::priority_queue<MergePair> pq;

  // Helper: add a merge candidate if valid
  auto tryAddMerge = [&](size_t pos) {
    if (pos >= tokens.size() - 1)
      return;

    std::string merged = tokens[pos] + tokens[pos + 1];
    if (Tokenizer::HasToken(model, merged)) {
      float score = Tokenizer::GetTokenScore(model, merged);
      pq.push({score, pos, generation[pos]});
    }
  };

  // Initialize: add all valid merge pairs
  for (size_t i = 0; i < tokens.size() - 1; ++i) {
    tryAddMerge(i);
  }

  // Process merges
  while (!pq.empty()) {
    MergePair top = pq.top();
    pq.pop();

    // Lazy deletion: skip if this entry is stale
    if (top.position >= tokens.size() - 1) {
      continue;
    }
    if (top.generation != generation[top.position]) {
      continue; // Position was invalidated by a previous merge
    }

    // Perform the merge
    size_t pos = top.position;
    std::string merged = tokens[pos] + tokens[pos + 1];

    // Verify merge is still valid (tokens might have changed)
    if (!Tokenizer::HasToken(model, merged)) {
      continue;
    }

    // Execute merge: replace tokens[pos] with merged, remove tokens[pos+1]
    tokens[pos] = merged;
    tokens.erase(tokens.begin() + pos + 1);

    // Update generation counters for affected positions
    if (pos > 0) {
      generation[pos - 1]++;
    }
    generation[pos]++;

    // Resize generation vector if needed
    if (generation.size() > tokens.size()) {
      generation.resize(tokens.size());
    }

    // Add new merge candidates for affected positions
    if (pos > 0) {
      tryAddMerge(pos - 1); // Left neighbor with new merged token
    }
    tryAddMerge(pos); // New merged token with right neighbor
  }

  return tokens;
}

// ============================================================================
// Tokenize
// ============================================================================

std::vector<int> Tokenizer::Tokenize(const TransformerModel *model,
                                     const std::string &text, bool add_bos,
                                     bool add_eos) {
  std::vector<int> result;

  // Add BOS if requested
  if (add_bos && model->bos_token_id >= 0) {
    result.push_back(model->bos_token_id);
  }

  if (text.empty()) {
    if (add_eos && model->eos_token_id >= 0) {
      result.push_back(model->eos_token_id);
    }
    return result;
  }

  // Step 1: Split into initial units (UTF-8 characters)
  std::vector<std::string> tokens = SplitToChars(text);

  // Step 2: BPE merge using optimized priority queue algorithm
  tokens = MergeWithPriorityQueue(model, std::move(tokens));

  // Step 3: Convert string tokens to IDs
  for (const std::string &tok : tokens) {
    auto it = model->token_to_id.find(tok);
    if (it != model->token_to_id.end()) {
      result.push_back(it->second);
    } else {
      // Byte fallback: encode each byte as <0xXX> or similar
      // For now, try to find individual characters/bytes
      for (size_t i = 0; i < tok.size(); i++) {
        // Try the single character
        std::string single_char = tok.substr(i, 1);
        auto c_it = model->token_to_id.find(single_char);
        if (c_it != model->token_to_id.end()) {
          result.push_back(c_it->second);
        } else {
          // Try byte token format: <0xXX>
          char buf[8];
          snprintf(buf, sizeof(buf), "<0x%02X>",
                   static_cast<unsigned char>(tok[i]));
          auto b_it = model->token_to_id.find(buf);
          if (b_it != model->token_to_id.end()) {
            result.push_back(b_it->second);
          }
          // If still not found, token is skipped (UNK handling varies by model)
        }
      }
    }
  }

  // Add EOS if requested
  if (add_eos && model->eos_token_id >= 0) {
    result.push_back(model->eos_token_id);
  }

  return result;
}

// ============================================================================
// Detokenize
// ============================================================================

std::string Tokenizer::Detokenize(const TransformerModel *model, int token_id) {
  if (token_id < 0 || token_id >= (int)model->vocab_tokens.size()) {
    return "";
  }

  std::string token = model->vocab_tokens[token_id];

  // Handle byte tokens: <0xXX> -> actual byte
  if (token.size() == 6 && token[0] == '<' && token[1] == '0' &&
      token[2] == 'x' && token[5] == '>') {
    char hex[3] = {token[3], token[4], 0};
    int byte_val = 0;
    if (sscanf(hex, "%x", &byte_val) == 1) {
      return std::string(1, static_cast<char>(byte_val));
    }
  }

  // Handle common BPE prefix characters
  // Many BPE tokenizers use "Ġ" (byte 0xC4 0xA0) to represent leading space
  // And "Ċ" for newline
  std::string result;
  result.reserve(token.size());

  for (size_t i = 0; i < token.size(); i++) {
    // Check for Ġ (U+0120 = 0xC4 0xA0) -> space
    if (i + 1 < token.size() && static_cast<unsigned char>(token[i]) == 0xC4 &&
        static_cast<unsigned char>(token[i + 1]) == 0xA0) {
      result += ' ';
      i++; // Skip next byte
    }
    // Check for Ċ (U+010A = 0xC4 0x8A) -> newline
    else if (i + 1 < token.size() &&
             static_cast<unsigned char>(token[i]) == 0xC4 &&
             static_cast<unsigned char>(token[i + 1]) == 0x8A) {
      result += '\n';
      i++;
    }
    // Check for ▁ (U+2581 = 0xE2 0x96 0x81) -> space (SentencePiece style)
    else if (i + 2 < token.size() &&
             static_cast<unsigned char>(token[i]) == 0xE2 &&
             static_cast<unsigned char>(token[i + 1]) == 0x96 &&
             static_cast<unsigned char>(token[i + 2]) == 0x81) {
      result += ' ';
      i += 2;
    } else {
      result += token[i];
    }
  }

  return result;
}

std::string Tokenizer::DetokenizeMultiple(const TransformerModel *model,
                                          const std::vector<int> &token_ids) {
  std::string result;
  for (int id : token_ids) {
    // Skip special tokens
    if (id == model->bos_token_id || id == model->eos_token_id) {
      continue;
    }
    result += Detokenize(model, id);
  }
  return result;
}
