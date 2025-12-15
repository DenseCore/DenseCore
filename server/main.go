package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"descore-server/internal/api"
	"descore-server/internal/config"
	"descore-server/internal/engine"
	"descore-server/internal/middleware"
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

	chatService := service.NewChatService(modelService)
	handler := api.NewHandler(chatService, modelService)

	// Initialize API key store
	var apiKeyStore middleware.APIKeyStore
	authEnabled := os.Getenv("AUTH_ENABLED") == "true"

	if authEnabled {
		slog.Info("authentication enabled")

		// Use in-memory key store
		keyStore := middleware.NewInMemoryKeyStore()
		apiKeysEnv := os.Getenv("API_KEYS")
		if apiKeysEnv != "" {
			loadAPIKeys(keyStore, apiKeysEnv)
		} else {
			keyStore.AddKey("sk-dev-test123", "dev-user", "enterprise")
			slog.Warn("using default development API key, set API_KEYS env for production")
		}
		apiKeyStore = keyStore
	} else {
		slog.Info("authentication disabled", slog.String("hint", "set AUTH_ENABLED=true to enable"))
	}

	// Create validation middleware
	validator := middleware.NewRequestValidator()
	_ = validator // Will be used in handlers

	// Create rate limiter (in-memory)
	var rateLimiter middleware.RateLimiterInterface
	if cfg.RateLimitEnabled {
		rateLimiter = middleware.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
	}
	// Create middleware chain
	var middlewareChain func(http.Handler) http.Handler

	if cfg.RateLimitEnabled && rateLimiter != nil {
		if authEnabled {
			middlewareChain = middleware.Chain(
				middleware.Recovery(),
				middleware.RequestID(),
				middleware.Tracing(),
				middleware.CORS(cfg.CORSAllowedOrigins),
				middleware.APIKeyAuth(apiKeyStore),
				middleware.RateLimitWithInterface(rateLimiter),
				middleware.MaxBodySize(cfg.MaxRequestBodySize),
				middleware.Logging(cfg.LogFormat),
			)
		} else {
			middlewareChain = middleware.Chain(
				middleware.Recovery(),
				middleware.RequestID(),
				middleware.Tracing(),
				middleware.CORS(cfg.CORSAllowedOrigins),
				middleware.RateLimitWithInterface(rateLimiter),
				middleware.MaxBodySize(cfg.MaxRequestBodySize),
				middleware.Logging(cfg.LogFormat),
			)
		}
	} else {
		if authEnabled {
			middlewareChain = middleware.Chain(
				middleware.Recovery(),
				middleware.RequestID(),
				middleware.Tracing(),
				middleware.CORS(cfg.CORSAllowedOrigins),
				middleware.APIKeyAuth(apiKeyStore),
				middleware.MaxBodySize(cfg.MaxRequestBodySize),
				middleware.Logging(cfg.LogFormat),
			)
		} else {
			middlewareChain = middleware.Chain(
				middleware.Recovery(),
				middleware.RequestID(),
				middleware.Tracing(),
				middleware.CORS(cfg.CORSAllowedOrigins),
				middleware.MaxBodySize(cfg.MaxRequestBodySize),
				middleware.Logging(cfg.LogFormat),
			)
		}
	}

	// Setup router
	mux := http.NewServeMux()

	// API endpoints
	mux.HandleFunc("/v1/chat/completions", handler.ChatCompletionHandler)
	mux.HandleFunc("/v1/embeddings", handler.EmbeddingsHandler)
	mux.HandleFunc("/v1/models", handler.ModelsHandler)
	mux.HandleFunc("/v1/models/load", handler.LoadModelHandler)
	mux.HandleFunc("/v1/models/unload", handler.UnloadModelHandler)

	// Health endpoints (K8s probes)
	mux.HandleFunc("/health", handler.HealthHandler)
	mux.HandleFunc("/health/live", handler.LivenessHandler)
	mux.HandleFunc("/health/ready", handler.ReadinessHandler)
	mux.HandleFunc("/health/startup", handler.StartupHandler)

	// Metrics endpoint
	mux.HandleFunc("/metrics", handler.MetricsHandler)

	// Root info endpoint
	mux.HandleFunc("/", rootHandler)

	// Apply middleware
	httpHandler := middlewareChain(mux)

	// Create server with timeouts
	server := &http.Server{
		Addr:         cfg.Address(),
		Handler:      httpHandler,
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
			server.Close()
		}

		// Cleanup resources
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

	authEnabled := os.Getenv("AUTH_ENABLED") == "true"

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"name":            "DenseCore API Server",
		"version":         version,
		"description":     "Cloud-native CPU inference engine with OpenAI-compatible API",
		"authentication":  authEnabled,
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


