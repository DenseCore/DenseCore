package middleware

import (
	"log/slog"
	"os"
)

// ============================================================================
// Global Logger Instance
// ============================================================================

var (
	// Logger is the global slog instance used by all middleware.
	// Configured for JSON output to stdout (production-ready for K8s).
	Logger *slog.Logger
)

func init() {
	// JSON output to stdout for structured logging (Datadog/Loki compatible)
	Logger = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	// Set as default logger for any log/slog calls
	slog.SetDefault(Logger)
}

// ============================================================================
// Request Logger (Context-Aware)
// ============================================================================

// RequestLogger provides structured logging with request context.
// Wraps slog.Logger with chained attribute building.
type RequestLogger struct {
	attrs []slog.Attr
}

// NewRequestLogger creates a new request logger.
func NewRequestLogger() *RequestLogger {
	return &RequestLogger{
		attrs: make([]slog.Attr, 0, 8),
	}
}

// Str adds a string field (chainable).
func (l *RequestLogger) Str(key, val string) *RequestLogger {
	l.attrs = append(l.attrs, slog.String(key, val))
	return l
}

// Int adds an integer field.
func (l *RequestLogger) Int(key string, val int) *RequestLogger {
	l.attrs = append(l.attrs, slog.Int(key, val))
	return l
}

// Int64 adds an int64 field.
func (l *RequestLogger) Int64(key string, val int64) *RequestLogger {
	l.attrs = append(l.attrs, slog.Int64(key, val))
	return l
}

// Float64 adds a float64 field.
func (l *RequestLogger) Float64(key string, val float64) *RequestLogger {
	l.attrs = append(l.attrs, slog.Float64(key, val))
	return l
}

// Err adds an error field.
func (l *RequestLogger) Err(err error) *RequestLogger {
	if err != nil {
		l.attrs = append(l.attrs, slog.String("error", err.Error()))
	}
	return l
}

// Any adds any value field.
func (l *RequestLogger) Any(key string, val any) *RequestLogger {
	l.attrs = append(l.attrs, slog.Any(key, val))
	return l
}

// toArgs converts attrs to []any for slog methods.
func (l *RequestLogger) toArgs() []any {
	args := make([]any, len(l.attrs))
	for i, attr := range l.attrs {
		args[i] = attr
	}
	return args
}

// Info logs at INFO level.
func (l *RequestLogger) Info(msg string) {
	Logger.Info(msg, l.toArgs()...)
}

// Debug logs at DEBUG level.
func (l *RequestLogger) Debug(msg string) {
	Logger.Debug(msg, l.toArgs()...)
}

// Warn logs at WARN level.
func (l *RequestLogger) Warn(msg string) {
	Logger.Warn(msg, l.toArgs()...)
}

// Error logs at ERROR level.
func (l *RequestLogger) Error(msg string) {
	Logger.Error(msg, l.toArgs()...)
}
