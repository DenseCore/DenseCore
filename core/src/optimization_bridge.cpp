#include "densecore.h"
#include "model_loader.h"
#include "pruner.h"
#include "quantizer.h"
#include "save_model.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// ============================================================================
// Lightweight Recursive-Descent JSON Parser (C++17)
// ============================================================================
// Supports: Objects, Arrays, Strings (with escapes), Numbers, Booleans, Null
// Memory-safe: Uses std::variant and RAII containers.
// ============================================================================

namespace {

class JsonValue {
public:
  using Object = std::unordered_map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;

  enum class Type { Null, Bool, Number, String, ArrayType, ObjectType };

private:
  Type type_ = Type::Null;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data_;

  // Sentinel for missing keys
  static const JsonValue &null_value() {
    static JsonValue null_val;
    return null_val;
  }

public:
  JsonValue() : type_(Type::Null), data_(nullptr) {}
  explicit JsonValue(std::nullptr_t) : type_(Type::Null), data_(nullptr) {}
  explicit JsonValue(bool v) : type_(Type::Bool), data_(v) {}
  explicit JsonValue(double v) : type_(Type::Number), data_(v) {}
  explicit JsonValue(const std::string &v) : type_(Type::String), data_(v) {}
  explicit JsonValue(std::string &&v)
      : type_(Type::String), data_(std::move(v)) {}
  explicit JsonValue(const Array &v) : type_(Type::ArrayType), data_(v) {}
  explicit JsonValue(Array &&v) : type_(Type::ArrayType), data_(std::move(v)) {}
  explicit JsonValue(const Object &v) : type_(Type::ObjectType), data_(v) {}
  explicit JsonValue(Object &&v)
      : type_(Type::ObjectType), data_(std::move(v)) {}

  Type type() const { return type_; }
  bool isNull() const { return type_ == Type::Null; }
  bool isObject() const { return type_ == Type::ObjectType; }
  bool isArray() const { return type_ == Type::ArrayType; }

  // Accessors with defaults (safe, no exceptions on type mismatch)
  std::string asString(const std::string &def = "") const {
    if (type_ == Type::String)
      return std::get<std::string>(data_);
    return def;
  }

  double asNumber(double def = 0.0) const {
    if (type_ == Type::Number)
      return std::get<double>(data_);
    return def;
  }

  int asInt(int def = 0) const {
    if (type_ == Type::Number)
      return static_cast<int>(std::get<double>(data_));
    return def;
  }

  bool asBool(bool def = false) const {
    if (type_ == Type::Bool)
      return std::get<bool>(data_);
    return def;
  }

  const Array &asArray() const {
    static const Array empty;
    if (type_ == Type::ArrayType)
      return std::get<Array>(data_);
    return empty;
  }

  const Object &asObject() const {
    static const Object empty;
    if (type_ == Type::ObjectType)
      return std::get<Object>(data_);
    return empty;
  }

  // Object member access: returns null_value() if key missing or not object
  const JsonValue &operator[](const std::string &key) const {
    if (type_ != Type::ObjectType)
      return null_value();
    const auto &obj = std::get<Object>(data_);
    auto it = obj.find(key);
    if (it == obj.end())
      return null_value();
    return it->second;
  }

  // ========================================================================
  // Parser Implementation
  // ========================================================================
  static JsonValue parse(const std::string &json) {
    const char *p = json.c_str();
    skipWhitespace(p);
    if (*p == '\0')
      return JsonValue();
    return parseValue(p);
  }

private:
  static void skipWhitespace(const char *&p) {
    while (*p && std::isspace(static_cast<unsigned char>(*p)))
      ++p;
  }

  static JsonValue parseValue(const char *&p) {
    skipWhitespace(p);
    if (*p == '\0')
      return JsonValue();

    switch (*p) {
    case '{':
      return parseObject(p);
    case '[':
      return parseArray(p);
    case '"':
      return JsonValue(parseString(p));
    case 't':
    case 'f':
    case 'n':
      return parseLiteral(p);
    default:
      if (*p == '-' || std::isdigit(static_cast<unsigned char>(*p))) {
        return parseNumber(p);
      }
      // Unknown token, return null
      return JsonValue();
    }
  }

  static JsonValue parseObject(const char *&p) {
    Object obj;
    ++p; // consume '{'
    skipWhitespace(p);

    if (*p == '}') {
      ++p;
      return JsonValue(std::move(obj));
    }

    while (true) {
      skipWhitespace(p);
      if (*p != '"')
        break; // malformed

      std::string key = parseString(p);
      skipWhitespace(p);
      if (*p != ':')
        break; // malformed
      ++p;     // consume ':'

      JsonValue val = parseValue(p);
      obj[std::move(key)] = std::move(val);

      skipWhitespace(p);
      if (*p == '}') {
        ++p;
        break;
      }
      if (*p == ',') {
        ++p;
        continue;
      }
      break; // malformed
    }
    return JsonValue(std::move(obj));
  }

  static JsonValue parseArray(const char *&p) {
    Array arr;
    ++p; // consume '['
    skipWhitespace(p);

    if (*p == ']') {
      ++p;
      return JsonValue(std::move(arr));
    }

    while (true) {
      arr.push_back(parseValue(p));
      skipWhitespace(p);
      if (*p == ']') {
        ++p;
        break;
      }
      if (*p == ',') {
        ++p;
        continue;
      }
      break; // malformed
    }
    return JsonValue(std::move(arr));
  }

  static std::string parseString(const char *&p) {
    std::string result;
    ++p; // consume opening '"'

    while (*p && *p != '"') {
      if (*p == '\\') {
        ++p;
        switch (*p) {
        case '"':
          result += '"';
          break;
        case '\\':
          result += '\\';
          break;
        case '/':
          result += '/';
          break;
        case 'b':
          result += '\b';
          break;
        case 'f':
          result += '\f';
          break;
        case 'n':
          result += '\n';
          break;
        case 'r':
          result += '\r';
          break;
        case 't':
          result += '\t';
          break;
        case 'u': {
          // Basic \uXXXX handling (ASCII range only for simplicity)
          if (p[1] && p[2] && p[3] && p[4]) {
            char hex[5] = {p[1], p[2], p[3], p[4], '\0'};
            unsigned int codepoint = std::strtoul(hex, nullptr, 16);
            if (codepoint < 0x80) {
              result += static_cast<char>(codepoint);
            } else if (codepoint < 0x800) {
              result += static_cast<char>(0xC0 | (codepoint >> 6));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            } else {
              result += static_cast<char>(0xE0 | (codepoint >> 12));
              result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            p += 4;
          }
          break;
        }
        default:
          result += *p;
          break;
        }
        ++p;
      } else {
        result += *p++;
      }
    }
    if (*p == '"')
      ++p; // consume closing '"'
    return result;
  }

  static JsonValue parseNumber(const char *&p) {
    const char *start = p;
    if (*p == '-')
      ++p;
    while (std::isdigit(static_cast<unsigned char>(*p)))
      ++p;
    if (*p == '.') {
      ++p;
      while (std::isdigit(static_cast<unsigned char>(*p)))
        ++p;
    }
    if (*p == 'e' || *p == 'E') {
      ++p;
      if (*p == '+' || *p == '-')
        ++p;
      while (std::isdigit(static_cast<unsigned char>(*p)))
        ++p;
    }
    std::string num_str(start, p);
    return JsonValue(std::strtod(num_str.c_str(), nullptr));
  }

  static JsonValue parseLiteral(const char *&p) {
    if (std::strncmp(p, "true", 4) == 0) {
      p += 4;
      return JsonValue(true);
    }
    if (std::strncmp(p, "false", 5) == 0) {
      p += 5;
      return JsonValue(false);
    }
    if (std::strncmp(p, "null", 4) == 0) {
      p += 4;
      return JsonValue();
    }
    return JsonValue();
  }
};

} // anonymous namespace

extern "C" {

int QuantizeModel(const char *model_path, const char *output_path,
                  const char *config_json) {
  if (!model_path || !output_path)
    return -1;

  try {
    std::string json_str = config_json ? config_json : "";
    auto cfg = JsonValue::parse(json_str);

    densecore::QuantConfig config;
    std::string fmt = cfg["format"].asString();
    if (fmt == "int8")
      config.format = densecore::QuantFormat::INT8;
    else if (fmt == "fp8_e4m3")
      config.format = densecore::QuantFormat::FP8_E4M3;
    else if (fmt == "int4" || fmt == "int4_blockwise")
      config.format = densecore::QuantFormat::INT4_BLOCKWISE;
    else
      config.format = densecore::QuantFormat::INT4_BLOCKWISE;

    std::string algo = cfg["algorithm"].asString();
    if (algo == "smoothquant")
      config.algorithm = densecore::QuantAlgorithm::SMOOTHQUANT;
    else if (algo == "awq_clip")
      config.algorithm = densecore::QuantAlgorithm::AWQ_CLIP;
    else if (algo == "max")
      config.algorithm = densecore::QuantAlgorithm::MAX;
    else
      config.algorithm = densecore::QuantAlgorithm::AWQ_LITE;

    config.block_size = cfg["block_size"].asInt(128);

    std::cout << "[Bridge] QuantizeModel: Loading " << model_path << "..."
              << std::endl;
    // Load model (weights only is fine for quantization typically, but we need
    // context)
    // Note: LoadModel loads into memory. We might need a streaming loader for
    // huge models.
    // For now, load fully.
    TransformerModel *model = LoadGGUFModel(model_path);
    if (!model)
      return -2;

    auto quantizer = densecore::CreateQuantizer(config);
    if (!quantizer) {
      delete model;
      return -3;
    }

    std::cout << "[Bridge] Starting quantization with "
              << config.GetAlgorithmName() << "..." << std::endl;
    // Iterate all tensors
    for (auto &layer : model->layers) {
      if (layer.wq)
        quantizer->QuantizeWeight(layer.wq);
      if (layer.wk)
        quantizer->QuantizeWeight(layer.wk);
      if (layer.wv)
        quantizer->QuantizeWeight(layer.wv);
      if (layer.wo)
        quantizer->QuantizeWeight(layer.wo);
      if (layer.w1)
        quantizer->QuantizeWeight(layer.w1);
      if (layer.w2)
        quantizer->QuantizeWeight(layer.w2);
      if (layer.w3)
        quantizer->QuantizeWeight(layer.w3);
    }
    // Also tok_embeddings and output? usually not quantized or different
    // config. Config controls skipping.

    std::cout << "[Bridge] Saving quantized model to " << output_path << "..."
              << std::endl;

    // Use the new SaveGGUFModel API
    if (!densecore::SaveGGUFModel(*model, output_path)) {
      std::cerr << "[Bridge] ERROR: Failed to save quantized model"
                << std::endl;
      delete model;
      return -4;
    }

    std::cout << "[Bridge] Quantization complete!" << std::endl;
    delete model;
    return 0; // Success!

  } catch (const std::exception &e) {
    std::cerr << "[Bridge] Exception: " << e.what() << std::endl;
    return -5;
  }
}

int PruneModel(const char *model_path, const char *output_path,
               const char *config_json) {
  if (!model_path || !output_path)
    return -1;

  try {
    std::string json_str = config_json ? config_json : "";
    auto cfg = JsonValue::parse(json_str);

    densecore::PruneConfig config;
    std::string strategy = cfg["strategy"].asString();
    if (strategy == "depth")
      config.strategy = densecore::PruneStrategy::DEPTH;
    else if (strategy == "width" || strategy == "width_structured")
      config.strategy = densecore::PruneStrategy::WIDTH;
    else if (strategy == "attention")
      config.strategy = densecore::PruneStrategy::ATTENTION;
    else if (strategy == "combined")
      config.strategy = densecore::PruneStrategy::COMBINED;
    else
      config.strategy = densecore::PruneStrategy::DEPTH; // Default to depth

    config.target_n_layer = cfg["target_n_layer"].asInt(0);
    config.target_hidden_size = cfg["target_hidden_size"].asInt(0);
    config.target_n_heads = cfg["target_n_heads"].asInt(0);
    config.target_ffn_hidden_size = cfg["target_ffn_hidden_size"].asInt(0);

    std::cout << "[Bridge] PruneModel: Loading " << model_path << "..."
              << std::endl;

    // Load model
    TransformerModel *model = LoadGGUFModel(model_path);
    if (!model)
      return -2;

    auto pruner = densecore::CreatePruner(config);
    if (!pruner) {
      delete model;
      return -3;
    }

    std::cout << "[Bridge] Starting " << config.GetStrategyName()
              << " pruning..." << std::endl;
    pruner->PruneModel(model);

    auto stats = pruner->GetStats();
    std::cout << "[Bridge] Pruning stats:" << std::endl;
    std::cout << "  Layer reduction: " << (stats.GetLayerReduction() * 100)
              << "%" << std::endl;
    std::cout << "  Width reduction: " << (stats.GetWidthReduction() * 100)
              << "%" << std::endl;

    std::cout << "[Bridge] Saving pruned model to " << output_path << "..."
              << std::endl;

    // Use the new SaveGGUFModel API
    if (!densecore::SaveGGUFModel(*model, output_path)) {
      std::cerr << "[Bridge] ERROR: Failed to save pruned model" << std::endl;
      delete model;
      return -4;
    }

    std::cout << "[Bridge] Pruning complete!" << std::endl;
    delete model;
    return 0; // Success!

  } catch (const std::exception &e) {
    std::cerr << "[Bridge] Exception: " << e.what() << std::endl;
    return -5;
  }
}
}
