package config

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Environment value constants to avoid magic strings
const envValueTrue = "true"

// ServerConfig holds all server configuration
type ServerConfig struct {
	// Server settings
	Port            int           `json:"port"`
	Host            string        `json:"host"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	ShutdownTimeout time.Duration `json:"shutdown_timeout"`

	// Rate Limiting
	RateLimitEnabled   bool `json:"rate_limit_enabled"`
	RateLimitReqPerSec int  `json:"rate_limit_rps"`
	RateLimitBurst     int  `json:"rate_limit_burst"`

	// Request settings
	MaxRequestBodySize int64         `json:"max_request_body_size"`
	RequestTimeout     time.Duration `json:"request_timeout"`

	// Model settings
	MainModelPath  string `json:"main_model_path"`
	DraftModelPath string `json:"draft_model_path"`
	Threads        int    `json:"threads"`

	// CPU Optimization
	EnableCPUAffinity bool `json:"cpu_affinity"`
	MaxConcurrency    int  `json:"max_concurrency"`

	// CORS
	CORSEnabled        bool     `json:"cors_enabled"`
	CORSAllowedOrigins []string `json:"cors_allowed_origins"`

	// Logging
	LogLevel  string `json:"log_level"`
	LogFormat string `json:"log_format"` // "json" or "text"

	// Metrics
	MetricsEnabled bool   `json:"metrics_enabled"`
	MetricsPath    string `json:"metrics_path"`
}

// DefaultConfig returns config with sensible defaults for production
func DefaultConfig() *ServerConfig {
	return &ServerConfig{
		Port:            8080,
		Host:            "0.0.0.0",
		ReadTimeout:     30 * time.Second,
		WriteTimeout:    120 * time.Second,
		IdleTimeout:     120 * time.Second,
		ShutdownTimeout: 30 * time.Second,

		RateLimitEnabled:   true,
		RateLimitReqPerSec: 100,
		RateLimitBurst:     200,

		MaxRequestBodySize: 10 * 1024 * 1024, // 10MB
		RequestTimeout:     120 * time.Second,

		Threads: 0, // Auto-detect

		EnableCPUAffinity: true,
		MaxConcurrency:    0, // Auto

		CORSEnabled:        true,
		CORSAllowedOrigins: []string{"*"},

		LogLevel:  "info",
		LogFormat: "json",

		MetricsEnabled: true,
		MetricsPath:    "/metrics",
	}
}

// LoadFromEnv loads configuration from environment variables
func LoadFromEnv() (*ServerConfig, error) {
	cfg := DefaultConfig()

	// Server
	if v := os.Getenv("PORT"); v != "" {
		if port, err := strconv.Atoi(v); err == nil {
			cfg.Port = port
		}
	}
	if v := os.Getenv("HOST"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("READ_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.ReadTimeout = d
		}
	}
	if v := os.Getenv("WRITE_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.WriteTimeout = d
		}
	}
	if v := os.Getenv("SHUTDOWN_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.ShutdownTimeout = d
		}
	}

	// Rate Limiting
	if v := os.Getenv("RATE_LIMIT_ENABLED"); v != "" {
		cfg.RateLimitEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("RATE_LIMIT_RPS"); v != "" {
		if rps, err := strconv.Atoi(v); err == nil {
			cfg.RateLimitReqPerSec = rps
		}
	}
	if v := os.Getenv("RATE_LIMIT_BURST"); v != "" {
		if burst, err := strconv.Atoi(v); err == nil {
			cfg.RateLimitBurst = burst
		}
	}

	// Model
	cfg.MainModelPath = os.Getenv("MAIN_MODEL_PATH")
	cfg.DraftModelPath = os.Getenv("DRAFT_MODEL_PATH")
	if v := os.Getenv("THREADS"); v != "" {
		if threads, err := strconv.Atoi(v); err == nil {
			cfg.Threads = threads
		}
	}

	// Auto-detect threads if not specified
	if cfg.Threads == 0 {
		cfg.Threads = runtime.NumCPU()
	}

	// CPU Optimization
	if v := os.Getenv("CPU_AFFINITY"); v != "" {
		cfg.EnableCPUAffinity = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("MAX_CONCURRENCY"); v != "" {
		if conc, err := strconv.Atoi(v); err == nil {
			cfg.MaxConcurrency = conc
		}
	}

	// CORS
	if v := os.Getenv("CORS_ENABLED"); v != "" {
		cfg.CORSEnabled = strings.ToLower(v) == envValueTrue || v == "1"
	}
	if v := os.Getenv("CORS_ORIGINS"); v != "" {
		cfg.CORSAllowedOrigins = strings.Split(v, ",")
	}

	// Logging
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("LOG_FORMAT"); v != "" {
		cfg.LogFormat = v
	}

	return cfg, nil
}

// Validate checks that the configuration is valid
func (c *ServerConfig) Validate() error {
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Port)
	}
	if c.Threads < 0 {
		return fmt.Errorf("invalid threads: %d", c.Threads)
	}
	if c.RateLimitReqPerSec < 0 {
		return fmt.Errorf("invalid rate limit: %d", c.RateLimitReqPerSec)
	}
	return nil
}

// Address returns the server address string
func (c *ServerConfig) Address() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

// String returns a human-readable config summary
func (c *ServerConfig) String() string {
	return fmt.Sprintf(
		"Config{addr=%s, threads=%d, rate_limit=%v(%d/s), timeout=%v}",
		c.Address(), c.Threads, c.RateLimitEnabled, c.RateLimitReqPerSec, c.RequestTimeout,
	)
}
