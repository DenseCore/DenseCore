package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"descore-server/internal/api"
	"descore-server/internal/config"
	"descore-server/internal/engine"
	"descore-server/internal/middleware"
	"descore-server/internal/queue"
	"descore-server/internal/service"
)

const (
	version = "2.0.0"
	banner  = `
==========================================
  DenseCore Server v%s
  Cloud-Native CPU Inference Engine
  OpenAI-Compatible REST API
  Production-Ready with Authentication
==========================================
`
	// envTrue is the expected value for boolean environment variables
	envTrue = "true"
)

func main() {
	// Initialize structured JSON logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	// Initialize OpenTelemetry propagator for distributed tracing
	middleware.InitOTelPropagator()

	fmt.Printf(banner, version)

	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("failed to load config", slog.String("error", err.Error()))
		os.Exit(1)
	}
	if err := cfg.Validate(); err != nil {
		slog.Error("invalid config", slog.String("error", err.Error()))
		os.Exit(1)
	}
	slog.Info("configuration loaded", slog.String("config", cfg.String()))

	// Detect and apply CPU configuration
	cpuCfg := engine.DetectCPUConfig()
	cpuCfg.Apply()

	// Override threads if config specifies
	threads := cfg.Threads
	if threads == 0 {
		threads = cpuCfg.OptimalThreadCount()
	}
	slog.Info("inference threads configured", slog.Int("threads", threads))

	// Initialize Services
	slog.Info("initializing services")
	modelService := service.NewModelService()

	// Load initial model in background (non-blocking startup)
	// This allows the HTTP server to start immediately for K8s probes
	if cfg.MainModelPath != "" {
		go func() {
			slog.Info("loading model in background", slog.String("model_path", cfg.MainModelPath))
			if err := modelService.LoadModel(cfg.MainModelPath, cfg.DraftModelPath, threads); err != nil {
				slog.Warn("failed to load model",
					slog.String("error", err.Error()),
					slog.String("hint", "use /v1/models/load to retry"),
				)
			} else {
				slog.Info("model loaded successfully, ready to serve requests")
			}
		}()
		slog.Info("model loading started in background, server will start immediately")
	} else {
		slog.Info("no model path specified, use /v1/models/load to load a model")
	}

	// Initialize Request Queue and Worker Pool
	// queueSize 1024 allows buffering bursts of traffic
	requestQueue := queue.NewRequestQueue(1024)

	// Start Worker Pool
	// We use 'threads' as the number of concurrent workers.
	// This ensures we don't oversubscribe the CPU if the engine handles concurrency,
	// or we process N requests in parallel.
	workerPool := service.NewQueueProcessor(requestQueue, modelService)
	workerPool.Start(threads)

	chatService := service.NewChatService(modelService, requestQueue)
	handler := api.NewHandler(chatService, modelService)

	// Initialize API key store
	var apiKeyStore middleware.APIKeyStore
	authEnabled := os.Getenv("AUTH_ENABLED") == envTrue

	if authEnabled {
		slog.Info("authentication enabled")

		// Try Redis key store first if configured
		redisURL := os.Getenv("REDIS_URL")
		redisKeystoreEnabled := os.Getenv("REDIS_KEYSTORE_ENABLED") == envTrue

		if redisURL != "" && redisKeystoreEnabled {
			cacheTTLStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_TTL", "5m")
			cacheTTL, err := time.ParseDuration(cacheTTLStr)
			if err != nil {
				slog.Warn("invalid REDIS_KEYSTORE_CACHE_TTL, using default",
					slog.String("value", cacheTTLStr),
					slog.String("default", "5m"),
					slog.String("error", err.Error()))
				cacheTTL = 5 * time.Minute
			}

			cacheSizeStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_SIZE", "1000")
			cacheSize, err := strconv.Atoi(cacheSizeStr)
			if err != nil {
				slog.Warn("invalid REDIS_KEYSTORE_CACHE_SIZE, using default",
					slog.String("value", cacheSizeStr),
					slog.String("default", "1000"),
					slog.String("error", err.Error()))
				cacheSize = 1000
			}

			redisDBStr := getEnvOrDefault("REDIS_DB", "0")
			redisDB, err := strconv.Atoi(redisDBStr)
			if err != nil {
				slog.Warn("invalid REDIS_DB, using default",
					slog.String("value", redisDBStr),
					slog.String("default", "0"),
					slog.String("error", err.Error()))
				redisDB = 0
			}

			redisKeyStore, err := middleware.NewRedisKeyStore(middleware.RedisKeyStoreConfig{
				RedisURL:      redisURL,
				RedisPassword: os.Getenv("REDIS_PASSWORD"),
				RedisDB:       redisDB,
				CacheTTL:      cacheTTL,
				CacheSize:     cacheSize,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for key store, falling back to in-memory",
					slog.String("error", err.Error()))
			} else {
				slog.Info("using Redis key store for distributed API key management")
				apiKeyStore = redisKeyStore
			}
		}

		// Fall back to in-memory if Redis not configured or failed
		if apiKeyStore == nil {
			keyStore := middleware.NewInMemoryKeyStore()
			apiKeysEnv := os.Getenv("API_KEYS")
			if apiKeysEnv != "" {
				loadAPIKeys(keyStore, apiKeysEnv)
			} else {
				keyStore.AddKey("sk-dev-test123", "dev-user", "enterprise")
				slog.Warn("using default development API key, set API_KEYS env for production")
			}
			apiKeyStore = keyStore
		}
	} else {
		slog.Info("authentication disabled", slog.String("hint", "set AUTH_ENABLED=true to enable"))
	}

	// Create validation middleware
	validator := middleware.NewRequestValidator()
	_ = validator // Will be used in handlers

	// Create rate limiter (Redis or in-memory)
	var rateLimiter middleware.RateLimiterInterface
	if cfg.RateLimitEnabled {
		redisURL := os.Getenv("REDIS_URL")
		redisRateLimitEnabled := os.Getenv("REDIS_RATELIMIT_ENABLED") == envTrue

		if redisURL != "" && redisRateLimitEnabled {
			redisDBStr := getEnvOrDefault("REDIS_DB", "0")
			redisDB, err := strconv.Atoi(redisDBStr)
			if err != nil {
				slog.Warn("invalid REDIS_DB for rate limiter, using default",
					slog.String("value", redisDBStr),
					slog.String("default", "0"),
					slog.String("error", err.Error()))
				redisDB = 0
			}

			redisRateLimiter, err := middleware.NewRedisRateLimiter(middleware.RedisRateLimiterConfig{
				RedisURL:          redisURL,
				RedisPassword:     os.Getenv("REDIS_PASSWORD"),
				RedisDB:           redisDB,
				RequestsPerSecond: cfg.RateLimitReqPerSec,
				Burst:             cfg.RateLimitBurst,
				FailureThreshold:  3,
				ResetTimeout:      30 * time.Second,
			})
			if err != nil {
				slog.Warn("failed to connect to Redis for rate limiting, falling back to in-memory",
					slog.String("error", err.Error()))
				rateLimiter = middleware.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
			} else {
				slog.Info("using Redis rate limiter for distributed rate limiting")
				rateLimiter = redisRateLimiter
			}
		} else {
			rateLimiter = middleware.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
		}
	}

	// 1. Global Chain: Applies to ALL requests
	//    - Recovery: Catch panics
	//    - RequestID: Tag requests
	//    - Tracing: Start OTel spans
	//    - Logging: Log access (skipped for health via exclusion later if needed, but good to check)
	globalChain := middleware.Chain(
		middleware.Recovery(),
		middleware.RequestID(),
		middleware.Tracing(), // Assuming Tracing middleware returns func(http.Handler) http.Handler
		middleware.Logging(cfg.LogFormat),
	)

	// 2. API Chain: Applies to /v1/ requests
	//    - RateLimit: Protect resources
	//    - CORS: Browser access
	//    - Auth: Security
	//    - MaxBodySize: DoS protection
	//    - ContentType: JSON default
	apiMiddleware := []func(http.Handler) http.Handler{}

	// Rate Limit
	if cfg.RateLimitEnabled && rateLimiter != nil {
		apiMiddleware = append(apiMiddleware, middleware.RateLimitWithInterface(rateLimiter))
	}

	// CORS (Positioned early to allow preflight, but after rate limit)
	apiMiddleware = append(apiMiddleware, middleware.CORS(cfg.CORSAllowedOrigins))

	// Authentication (If enabled)
	if authEnabled {
		apiMiddleware = append(apiMiddleware, middleware.APIKeyAuth(apiKeyStore))
	}

	// Safety & Defaults
	apiMiddleware = append(apiMiddleware,
		middleware.MaxBodySize(cfg.MaxRequestBodySize),
		middleware.ContentType("application/json"),
	)

	apiChain := middleware.Chain(apiMiddleware...)

	// ========================================================================
	// Routing Strategy
	// ========================================================================

	rootMux := http.NewServeMux()

	// --- 1. Health & Custom (Protected only by Global Chain) ---
	// We use a separate mux for health to avoid API middleware (Auth/RateLimit)
	healthMux := http.NewServeMux()
	healthMux.HandleFunc("/health", handler.HealthHandler)
	healthMux.HandleFunc("/health/live", handler.LivenessHandler)
	healthMux.HandleFunc("/health/ready", handler.ReadinessHandler)
	healthMux.HandleFunc("/health/startup", handler.StartupHandler)
	healthMux.HandleFunc("/metrics", handler.MetricsHandler)
	healthMux.HandleFunc("/", rootHandler) // Root info

	// Mount Health on Root
	// Note: We mount individual paths or the prefix.
	// Since /health is a prefix for others, we can handle it specifically.
	rootMux.Handle("/health", healthMux)
	rootMux.Handle("/health/", healthMux)
	rootMux.Handle("/metrics", healthMux)
	rootMux.Handle("/", healthMux) // Root handler fallback

	// --- 2. API V1 (Protected by API Chain + Global Chain) ---
	apiMux := http.NewServeMux()
	apiMux.HandleFunc("/v1/chat/completions", handler.ChatCompletionHandler)
	apiMux.HandleFunc("/v1/embeddings", handler.EmbeddingsHandler)
	apiMux.HandleFunc("/v1/models", handler.ModelsHandler)
	apiMux.HandleFunc("/v1/models/load", handler.LoadModelHandler)
	apiMux.HandleFunc("/v1/models/unload", handler.UnloadModelHandler)

	// Wrap API mux with API Chain
	apiHandler := apiChain(apiMux)

	// Mount API on Root
	rootMux.Handle("/v1/", apiHandler)

	// Wrap Root with Global Chain
	finalHandler := globalChain(rootMux)

	// Create server with timeouts
	server := &http.Server{
		Addr:         cfg.Address(),
		Handler:      finalHandler,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// Start server in goroutine
	serverErrors := make(chan error, 1)
	go func() {
		slog.Info("starting server",
			slog.String("address", cfg.Address()),
			slog.Bool("auth_enabled", authEnabled),
		)
		serverErrors <- server.ListenAndServe()
	}()

	// Wait for shutdown signal
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-serverErrors:
		slog.Error("server error", slog.String("error", err.Error()))
		os.Exit(1)

	case sig := <-shutdown:
		slog.Info("received shutdown signal, starting graceful shutdown", slog.String("signal", sig.String()))

		// Create shutdown context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()

		// Attempt graceful shutdown
		if err := server.Shutdown(ctx); err != nil {
			slog.Error("graceful shutdown failed, forcing close", slog.String("error", err.Error()))
			if closeErr := server.Close(); closeErr != nil {
				slog.Error("failed to close server", slog.String("error", closeErr.Error()))
			}
		}

		// Cleanup resources
		slog.Info("stopping worker pool")
		workerPool.Stop()

		slog.Info("unloading model")
		if err := modelService.UnloadModel(); err != nil {
			slog.Error("failed to unload model", slog.String("error", err.Error()))
		}

		slog.Info("shutdown complete")
	}
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	authEnabled := os.Getenv("AUTH_ENABLED") == envTrue

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"name":           "DenseCore API Server",
		"version":        version,
		"description":    "Cloud-native CPU inference engine with OpenAI-compatible API",
		"authentication": authEnabled,
		"endpoints": map[string]string{
			"chat":       "/v1/chat/completions",
			"embeddings": "/v1/embeddings",
			"models":     "/v1/models",
			"health":     "/health",
			"metrics":    "/metrics",
		},
		"health_probes": map[string]string{
			"liveness":  "/health/live",
			"readiness": "/health/ready",
			"startup":   "/health/startup",
		},
	})
}

// loadAPIKeys parses API keys from environment variable
// Format: key1:user1:tier1,key2:user2:tier2
func loadAPIKeys(store *middleware.InMemoryKeyStore, apiKeysEnv string) {
	if apiKeysEnv == "" {
		return
	}

	keys := strings.Split(apiKeysEnv, ",")
	for _, keyData := range keys {
		parts := strings.Split(strings.TrimSpace(keyData), ":")
		if len(parts) < 3 {
			slog.Warn("invalid API key format", slog.String("hint", "expected format: key:user:tier"))
			continue
		}

		key := strings.TrimSpace(parts[0])
		userID := strings.TrimSpace(parts[1])
		tier := strings.TrimSpace(parts[2])

		store.AddKey(key, userID, tier)
		slog.Info("loaded API key", slog.String("user_id", userID), slog.String("tier", tier))
	}
}

// getEnvOrDefault returns environment variable value or default if not set.
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
