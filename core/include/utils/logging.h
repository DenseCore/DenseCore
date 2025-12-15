#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace densecore {
namespace logging {

/**
 * Log severity levels.
 */
enum class Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

/**
 * Thread-safe logger singleton.
 *
 * Usage:
 *   LOG_INFO("Processing request ", request_id);
 *   LOG_ERROR("Failed to allocate memory: ", error_msg);
 */
class Logger {
public:
  /**
   * Get global logger instance.
   */
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  /**
   * Set minimum log level (messages below this are ignored).
   */
  void setLevel(Level level) {
    std::lock_guard<std::mutex> lock(mu_);
    min_level_ = level;
  }

  /**
   * Get current log level.
   */
  Level getLevel() const { return min_level_; }

  /**
   * Log a message with variadic arguments.
   */
  template <typename... Args>
  void log(Level level, const char *, int line, const char *func,
           Args &&...args) {
    if (level < min_level_)
      return;

    std::ostringstream oss;

    // Format: [LEVEL] [timestamp] [function:line] message
    oss << "[" << levelToString(level) << "] "
        << "[" << getTimestamp() << "] "
        << "[" << func << ":" << line << "] ";

    // Concatenate all arguments
    (oss << ... << args);
    oss << std::endl;

    // Thread-safe output
    std::lock_guard<std::mutex> lock(mu_);
    if (level >= Level::ERROR) {
      std::cerr << oss.str() << std::flush;
    } else {
      std::cout << oss.str() << std::flush;
    }
  }

  /**
   * Enable/disable logging.
   */
  void setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mu_);
    enabled_ = enabled;
  }

  bool isEnabled() const { return enabled_; }

private:
  Logger() = default;

  // Non-copyable
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  Level min_level_ = Level::INFO;
  bool enabled_ = true;
  std::mutex mu_;

  /**
   * Convert log level to string.
   */
  static const char *levelToString(Level level) {
    switch (level) {
    case Level::DEBUG:
      return "DEBUG  ";
    case Level::INFO:
      return "INFO   ";
    case Level::WARNING:
      return "WARNING";
    case Level::ERROR:
      return "ERROR  ";
    case Level::FATAL:
      return "FATAL  ";
    default:
      return "UNKNOWN";
    }
  }

  /**
   * Get current timestamp as string.
   */
  static std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
  }
};

} // namespace logging
} // namespace densecore

/**
 * Convenient logging macros with automatic location info.
 */
#define LOG_DEBUG(...)                                                         \
  densecore::logging::Logger::getInstance().log(                               \
      densecore::logging::Level::DEBUG, __FILE__, __LINE__, __FUNCTION__,      \
      __VA_ARGS__)

#define LOG_INFO(...)                                                          \
  densecore::logging::Logger::getInstance().log(                               \
      densecore::logging::Level::INFO, __FILE__, __LINE__, __FUNCTION__,       \
      __VA_ARGS__)

#define LOG_WARNING(...)                                                       \
  densecore::logging::Logger::getInstance().log(                               \
      densecore::logging::Level::WARNING, __FILE__, __LINE__, __FUNCTION__,    \
      __VA_ARGS__)

#define LOG_ERROR(...)                                                         \
  densecore::logging::Logger::getInstance().log(                               \
      densecore::logging::Level::ERROR, __FILE__, __LINE__, __FUNCTION__,      \
      __VA_ARGS__)

#define LOG_FATAL(...)                                                         \
  densecore::logging::Logger::getInstance().log(                               \
      densecore::logging::Level::FATAL, __FILE__, __LINE__, __FUNCTION__,      \
      __VA_ARGS__)
