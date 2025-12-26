package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	Version = "2.0.0"
	Banner  = `
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

// Options for starting the server
type Options struct {
	Host         string
	Port         int
	ModelPath    string
	Verbose      bool
	LogOutput    io.Writer // If nil, defaults to os.Stdout. Use io.Discard to silence logs.
	ShowBanner   bool      // Whether to print the ASCII banner
	Background   bool      // If true, don't wait for shutdown signals (caller manages lifecycle)
	ShutdownChan chan struct{} // Channel to signal shutdown in background mode
}

// ServerInstance represents a running server instance for external control
type ServerInstance struct {
	httpServer   *http.Server
	workerPool   *service.QueueProcessor
	modelService *service.ModelService
	shutdownChan chan struct{}
}

// Shutdown gracefully shuts down the server
func (s *ServerInstance) Shutdown(ctx context.Context) error {
	if err := s.httpServer.Shutdown(ctx); err != nil {
		return err
	}
	s.workerPool.Stop()
	return s.modelService.UnloadModel()
}

// Run starts the DenseCore server (blocking mode for CLI `serve` command)
func Run(opts *Options) error {
	instance, err := Start(opts)
	if err != nil {
		return err
	}

	if opts.Background {
		// In background mode, wait for shutdown signal from caller
		<-opts.ShutdownChan
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return instance.Shutdown(ctx)
	}

	// Wait for OS shutdown signal
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)
	<-shutdown

	slog.Info("received shutdown signal, starting graceful shutdown")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return instance.Shutdown(ctx)
}

// Start initializes and starts the server, returning a controllable instance
func Start(opts *Options) (*ServerInstance, error) {
	// Configure logging output
	logOutput := opts.LogOutput
	if logOutput == nil {
		logOutput = os.Stdout
	}

	logLevel := slog.LevelInfo
	if opts.Verbose {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewJSONHandler(logOutput, &slog.HandlerOptions{
		Level: logLevel,
	}))
	slog.SetDefault(logger)

	// Initialize OpenTelemetry propagator for distributed tracing
	middleware.InitOTelPropagator()

	if opts.ShowBanner {
		fmt.Printf(Banner, Version)
	}

	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("failed to load config", slog.String("error", err.Error()))
		return nil, err
	}

	// Override config with CLI options
	if opts.Host != "" {
		cfg.Host = opts.Host
	}
	if opts.Port != 0 {
		cfg.Port = opts.Port
	}
	if opts.ModelPath != "" {
		cfg.MainModelPath = opts.ModelPath
	}

	if err := cfg.Validate(); err != nil {
		slog.Error("invalid config", slog.String("error", err.Error()))
		return nil, err
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
	requestQueue := queue.NewRequestQueue(1024)
	workerPool := service.NewQueueProcessor(requestQueue, modelService)
	workerPool.Start(threads)

	chatService := service.NewChatService(modelService, requestQueue)
	handler := api.NewHandler(chatService, modelService)

	// Initialize API key store
	var apiKeyStore middleware.APIKeyStore
	authEnabled := os.Getenv("AUTH_ENABLED") == envTrue

	if authEnabled {
		slog.Info("authentication enabled")
		redisURL := os.Getenv("REDIS_URL")
		redisKeystoreEnabled := os.Getenv("REDIS_KEYSTORE_ENABLED") == envTrue

		if redisURL != "" && redisKeystoreEnabled {
			cacheTTLStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_TTL", "5m")
			cacheTTL, err := time.ParseDuration(cacheTTLStr)
			if err != nil {
				cacheTTL = 5 * time.Minute
			}

			cacheSizeStr := getEnvOrDefault("REDIS_KEYSTORE_CACHE_SIZE", "1000")
			cacheSize, err := strconv.Atoi(cacheSizeStr)
			if err != nil {
				cacheSize = 1000
			}

			redisDBStr := getEnvOrDefault("REDIS_DB", "0")
			redisDB, err := strconv.Atoi(redisDBStr)
			if err != nil {
				redisDB = 0
			}

			redisKeyStore, err := middleware.NewRedisKeyStore(middleware.RedisKeyStoreConfig{
				RedisURL:      redisURL,
				RedisPassword: os.Getenv("REDIS_PASSWORD"),
				RedisDB:       redisDB,
				CacheTTL:      cacheTTL,
				CacheSize:     cacheSize,
			})
			if err == nil {
				slog.Info("using Redis key store for distributed API key management")
				apiKeyStore = redisKeyStore
			}
		}

		if apiKeyStore == nil {
			keyStore := middleware.NewInMemoryKeyStore()
			apiKeysEnv := os.Getenv("API_KEYS")
			if apiKeysEnv != "" {
				loadAPIKeys(keyStore, apiKeysEnv)
			} else {
				keyStore.AddKey("sk-dev-test123", "dev-user", "enterprise")
			}
			apiKeyStore = keyStore
		}
	}

	// Create rate limiter
	var rateLimiter middleware.RateLimiterInterface
	if cfg.RateLimitEnabled {
		redisURL := os.Getenv("REDIS_URL")
		redisRateLimitEnabled := os.Getenv("REDIS_RATELIMIT_ENABLED") == envTrue

		if redisURL != "" && redisRateLimitEnabled {
			redisDBStr := getEnvOrDefault("REDIS_DB", "0")
			redisDB, _ := strconv.Atoi(redisDBStr)

			redisRateLimiter, err := middleware.NewRedisRateLimiter(middleware.RedisRateLimiterConfig{
				RedisURL:          redisURL,
				RedisPassword:     os.Getenv("REDIS_PASSWORD"),
				RedisDB:           redisDB,
				RequestsPerSecond: cfg.RateLimitReqPerSec,
				Burst:             cfg.RateLimitBurst,
				FailureThreshold:  3,
				ResetTimeout:      30 * time.Second,
			})
			if err == nil {
				rateLimiter = redisRateLimiter
			} else {
				rateLimiter = middleware.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
			}
		} else {
			rateLimiter = middleware.NewRateLimiter(cfg.RateLimitReqPerSec, cfg.RateLimitBurst)
		}
	}

	// Build middleware chains
	globalChain := middleware.Chain(
		middleware.Recovery(),
		middleware.RequestID(),
		middleware.Tracing(),
		middleware.Logging(cfg.LogFormat),
	)

	apiMiddleware := []func(http.Handler) http.Handler{}
	if cfg.RateLimitEnabled && rateLimiter != nil {
		apiMiddleware = append(apiMiddleware, middleware.RateLimitWithInterface(rateLimiter))
	}
	apiMiddleware = append(apiMiddleware, middleware.CORS(cfg.CORSAllowedOrigins))
	if authEnabled {
		apiMiddleware = append(apiMiddleware, middleware.APIKeyAuth(apiKeyStore))
	}
	apiMiddleware = append(apiMiddleware,
		middleware.MaxBodySize(cfg.MaxRequestBodySize),
		middleware.ContentType("application/json"),
	)
	apiChain := middleware.Chain(apiMiddleware...)

	// Setup routes
	rootMux := http.NewServeMux()

	healthMux := http.NewServeMux()
	healthMux.HandleFunc("/health", handler.HealthHandler)
	healthMux.HandleFunc("/health/live", handler.LivenessHandler)
	healthMux.HandleFunc("/health/ready", handler.ReadinessHandler)
	healthMux.HandleFunc("/health/startup", handler.StartupHandler)
	healthMux.HandleFunc("/metrics", handler.MetricsHandler)
	healthMux.HandleFunc("/", rootHandler)

	rootMux.Handle("/health", healthMux)
	rootMux.Handle("/health/", healthMux)
	rootMux.Handle("/metrics", healthMux)
	rootMux.Handle("/", healthMux)

	apiMux := http.NewServeMux()
	apiMux.HandleFunc("/v1/chat/completions", handler.ChatCompletionHandler)
	apiMux.HandleFunc("/v1/embeddings", handler.EmbeddingsHandler)
	apiMux.HandleFunc("/v1/models", handler.ModelsHandler)
	apiMux.HandleFunc("/v1/models/load", handler.LoadModelHandler)
	apiMux.HandleFunc("/v1/models/unload", handler.UnloadModelHandler)

	apiHandler := apiChain(apiMux)
	rootMux.Handle("/v1/", apiHandler)

	finalHandler := globalChain(rootMux)

	// Create server
	httpServer := &http.Server{
		Addr:         fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler:      finalHandler,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// Start server in goroutine
	go func() {
		slog.Info("starting server", slog.String("address", httpServer.Addr))
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", slog.String("error", err.Error()))
		}
	}()

	return &ServerInstance{
		httpServer:   httpServer,
		workerPool:   workerPool,
		modelService: modelService,
		shutdownChan: opts.ShutdownChan,
	}, nil
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
		"version":        Version,
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

func loadAPIKeys(store *middleware.InMemoryKeyStore, apiKeysEnv string) {
	if apiKeysEnv == "" {
		return
	}

	keys := strings.Split(apiKeysEnv, ",")
	for _, keyData := range keys {
		parts := strings.Split(strings.TrimSpace(keyData), ":")
		if len(parts) < 3 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		userID := strings.TrimSpace(parts[1])
		tier := strings.TrimSpace(parts[2])

		store.AddKey(key, userID, tier)
		slog.Info("loaded API key", slog.String("user_id", userID), slog.String("tier", tier))
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
