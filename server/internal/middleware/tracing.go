package middleware

import (
	"context"
	"net/http"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// ============================================================================
// OpenTelemetry Context Helpers
// ============================================================================

const (
	TraceIDKey      contextKey = "trace_id"
	SpanIDKey       contextKey = "span_id"
	ParentSpanIDKey contextKey = "parent_span_id"
)

// TraceInfo holds distributed tracing information
type TraceInfo struct {
	TraceID      string    `json:"trace_id"`
	SpanID       string    `json:"span_id"`
	ParentSpanID string    `json:"parent_span_id,omitempty"`
	StartTime    time.Time `json:"start_time"`
}

// GetTraceID extracts the trace ID from the OTel span context.
// Returns empty string if no active span.
func GetTraceID(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if span.SpanContext().HasTraceID() {
		return span.SpanContext().TraceID().String()
	}
	// Fallback to request ID if no OTel span
	return GetRequestID(ctx)
}

// GetSpanID extracts the span ID from the OTel span context.
func GetSpanID(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if span.SpanContext().HasSpanID() {
		return span.SpanContext().SpanID().String()
	}
	return ""
}

// ============================================================================
// OpenTelemetry Tracing Middleware
// ============================================================================

// Tracing creates OpenTelemetry-compatible tracing middleware.
// Extracts incoming trace context from W3C Trace Context headers (traceparent).
// Creates a new span for each request and injects trace context into responses.
func Tracing() func(http.Handler) http.Handler {
	tracer := otel.Tracer("densecore-server")
	propagator := otel.GetTextMapPropagator()

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Extract incoming trace context from headers
			ctx := propagator.Extract(r.Context(), propagation.HeaderCarrier(r.Header))

			// Start a new span for this request
			ctx, span := tracer.Start(ctx, r.Method+" "+r.URL.Path)
			defer span.End()

			// Inject trace context into response headers for downstream correlation
			propagator.Inject(ctx, propagation.HeaderCarrier(w.Header()))

			// Also set legacy headers for backward compatibility
			if span.SpanContext().HasTraceID() {
				w.Header().Set("X-Trace-ID", span.SpanContext().TraceID().String())
			}
			if span.SpanContext().HasSpanID() {
				w.Header().Set("X-Span-ID", span.SpanContext().SpanID().String())
			}

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// ============================================================================
// OpenTelemetry Initialization (called from main.go)
// ============================================================================

// InitOTelPropagator sets up the global text map propagator.
// Call this during server initialization.
func InitOTelPropagator() {
	// Use W3C Trace Context for distributed tracing
	otel.SetTextMapPropagator(propagation.TraceContext{})
}

// ============================================================================
// Request Metrics
// ============================================================================

// RequestMetrics tracks per-request timing
type RequestMetrics struct {
	StartTime     time.Time
	FirstByteTime time.Time
	EndTime       time.Time
	StatusCode    int
	BytesWritten  int64
}

// Duration returns total request duration
func (m *RequestMetrics) Duration() time.Duration {
	return m.EndTime.Sub(m.StartTime)
}

// TimeToFirstByte returns time to first byte
func (m *RequestMetrics) TimeToFirstByte() time.Duration {
	if m.FirstByteTime.IsZero() {
		return 0
	}
	return m.FirstByteTime.Sub(m.StartTime)
}

// metricsWriter wraps ResponseWriter to capture metrics
type metricsWriter struct {
	http.ResponseWriter
	metrics *RequestMetrics
}

func (mw *metricsWriter) WriteHeader(code int) {
	if mw.metrics.FirstByteTime.IsZero() {
		mw.metrics.FirstByteTime = time.Now()
	}
	mw.metrics.StatusCode = code
	mw.ResponseWriter.WriteHeader(code)
}

func (mw *metricsWriter) Write(b []byte) (int, error) {
	if mw.metrics.FirstByteTime.IsZero() {
		mw.metrics.FirstByteTime = time.Now()
	}
	n, err := mw.ResponseWriter.Write(b)
	mw.metrics.BytesWritten += int64(n)
	return n, err
}
