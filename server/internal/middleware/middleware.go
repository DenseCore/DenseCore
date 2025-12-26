package middleware

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ============================================================================
// Request ID Middleware
// ============================================================================

// RequestIDKey is the context key for request ID
type contextKey string

const RequestIDKey contextKey = "request_id"

// RequestID adds a unique request ID to each request
func RequestID() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Check for existing request ID in header
			requestID := r.Header.Get("X-Request-ID")
			if requestID == "" {
				requestID = uuid.New().String()
			}

			// Add to response header
			w.Header().Set("X-Request-ID", requestID)

			// Add to context
			ctx := context.WithValue(r.Context(), RequestIDKey, requestID)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// GetRequestID extracts request ID from context
func GetRequestID(ctx context.Context) string {
	if id, ok := ctx.Value(RequestIDKey).(string); ok {
		return id
	}
	return ""
}

// ============================================================================
// Rate Limiting Middleware (Token Bucket)
// ============================================================================

// RateLimiterInterface defines the contract for rate limiters.
// Both in-memory and Redis implementations satisfy this interface.
type RateLimiterInterface interface {
	Allow() bool
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(requestsPerSecond int, burst int) *RateLimiter {
	return &RateLimiter{
		tokens:     float64(burst),
		maxTokens:  float64(burst),
		refillRate: float64(requestsPerSecond),
		lastRefill: time.Now(),
	}
}

// Allow checks if a request is allowed
func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(rl.lastRefill).Seconds()
	rl.tokens += elapsed * rl.refillRate
	if rl.tokens > rl.maxTokens {
		rl.tokens = rl.maxTokens
	}
	rl.lastRefill = now

	if rl.tokens >= 1 {
		rl.tokens--
		return true
	}
	return false
}

// RateLimit creates rate limiting middleware
func RateLimit(limiter *RateLimiter) func(http.Handler) http.Handler {
	return RateLimitWithInterface(limiter)
}

// RateLimitWithInterface creates rate limiting middleware with any RateLimiterInterface.
// Use this for Redis or other distributed rate limiters.
func RateLimitWithInterface(limiter RateLimiterInterface) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !limiter.Allow() {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"error": map[string]string{
						"message": "Rate limit exceeded",
						"type":    "rate_limit_error",
						"code":    "rate_limit_exceeded",
					},
				})
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// ============================================================================
// Request Timeout Middleware
// ============================================================================

// ============================================================================
// Recovery Middleware (Panic Handler)
// ============================================================================

// Recovery catches panics and returns 500
func Recovery() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					requestID := GetRequestID(r.Context())
					Logger.Error("panic recovered",
						slog.String("request_id", requestID),
						slog.Any("error", err),
						slog.String("stack", string(debug.Stack())),
					)

					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					_ = json.NewEncoder(w).Encode(map[string]interface{}{
						"error": map[string]string{
							"message": "Internal server error",
							"type":    "internal_error",
							"code":    "internal_server_error",
						},
					})
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// ============================================================================
// CORS Middleware
// ============================================================================

// CORS adds CORS headers
func CORS(allowedOrigins []string) func(http.Handler) http.Handler {
	allowAll := len(allowedOrigins) == 1 && allowedOrigins[0] == "*"

	isAllowed := func(origin string) bool {
		if allowAll {
			return true
		}
		for _, allowed := range allowedOrigins {
			if origin == allowed {
				return true
			}
		}
		return false
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			if origin != "" && isAllowed(origin) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-ID")
				w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID")
				w.Header().Set("Access-Control-Max-Age", "86400")
			}

			// Handle preflight
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// ============================================================================
// Logging Middleware
// ============================================================================

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	written    int64
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.written += int64(n)
	return n, err
}

// Logging logs request details using slog (structured JSON).
func Logging(logFormat string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			requestID := GetRequestID(r.Context())

			// Wrap response writer
			wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

			next.ServeHTTP(wrapped, r)

			duration := time.Since(start)
			latencyMs := float64(duration.Microseconds()) / 1000.0

			// Extract client IP (handle X-Forwarded-For for proxied requests)
			clientIP := r.RemoteAddr
			if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
				// Take the first IP in the chain
				if idx := strings.Index(forwarded, ","); idx != -1 {
					clientIP = strings.TrimSpace(forwarded[:idx])
				} else {
					clientIP = strings.TrimSpace(forwarded)
				}
			}

			// Build structured log attributes
			attrs := []any{
				slog.String("request_id", requestID),
				slog.String("trace_id", GetTraceID(r.Context())),
				slog.String("method", r.Method),
				slog.String("path", r.URL.Path),
				slog.Int("status_code", wrapped.statusCode),
				slog.Float64("latency_ms", latencyMs),
				slog.String("client_ip", clientIP),
				slog.String("user_agent", r.UserAgent()),
				slog.Int64("bytes_written", wrapped.written),
			}

			// Log at appropriate level based on status code
			switch {
			case wrapped.statusCode >= 500:
				Logger.Error("request completed", attrs...)
			case wrapped.statusCode >= 400:
				Logger.Warn("request completed", attrs...)
			default:
				Logger.Info("request completed", attrs...)
			}
		})
	}
}

// ============================================================================
// Max Body Size Middleware
// ============================================================================

// MaxBodySize limits request body size
func MaxBodySize(maxBytes int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.ContentLength > maxBytes {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusRequestEntityTooLarge)
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"error": map[string]string{
						"message": "Request body too large",
						"type":    "invalid_request_error",
						"code":    "request_too_large",
					},
				})
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
			next.ServeHTTP(w, r)
		})
	}
}

// ============================================================================
// Chain combines multiple middleware
// ============================================================================

// Chain combines middleware in order
func Chain(middlewares ...func(http.Handler) http.Handler) func(http.Handler) http.Handler {
	return func(final http.Handler) http.Handler {
		for i := len(middlewares) - 1; i >= 0; i-- {
			final = middlewares[i](final)
		}
		return final
	}
}

// ============================================================================
// Content Type Middleware
// ============================================================================

// ContentType sets default content type for API responses
func ContentType(contentType string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Only set if not already set and path starts with /v1/
			if strings.HasPrefix(r.URL.Path, "/v1/") {
				if w.Header().Get("Content-Type") == "" {
					w.Header().Set("Content-Type", contentType)
				}
			}
			next.ServeHTTP(w, r)
		})
	}
}
