package middleware

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// tokenBucketScript is the Lua script for atomic token bucket operations.
// Returns 1 if request is allowed, 0 if denied.
const tokenBucketScript = `
local key = KEYS[1]
local maxTokens = tonumber(ARGV[1])
local refillRate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'lastRefill')
local tokens = tonumber(data[1]) or maxTokens
local lastRefill = tonumber(data[2]) or now

local elapsed = now - lastRefill
tokens = math.min(maxTokens, tokens + elapsed * refillRate)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
    redis.call('EXPIRE', key, 3600)
    return 1
else
    redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
    redis.call('EXPIRE', key, 3600)
    return 0
end
`

// RedisRateLimiter provides distributed token bucket rate limiting.
// Falls back to in-memory limiting if Redis is unavailable (Circuit Breaker pattern).
type RedisRateLimiter struct {
	client     *redis.Client
	fallback   *RateLimiter
	script     *redis.Script
	keyPrefix  string
	maxTokens  int
	refillRate int

	// Circuit breaker state
	mu           sync.RWMutex
	failures     int
	lastFailure  time.Time
	circuitOpen  bool
	threshold    int           // failures before opening circuit
	resetTimeout time.Duration // time before retrying Redis
}

// RedisRateLimiterConfig holds configuration for the Redis rate limiter.
type RedisRateLimiterConfig struct {
	RedisURL      string
	RedisPassword string
	RedisDB       int
	RedisTLS      bool

	RequestsPerSecond int
	Burst             int

	// Circuit breaker settings
	FailureThreshold int           // Default: 3
	ResetTimeout     time.Duration // Default: 30s
}

// NewRedisRateLimiter creates a new distributed rate limiter.
// Returns an error if Redis connection fails (caller should fall back to in-memory).
func NewRedisRateLimiter(cfg RedisRateLimiterConfig) (*RedisRateLimiter, error) {
	opts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return nil, err
	}

	if cfg.RedisPassword != "" {
		opts.Password = cfg.RedisPassword
	}
	opts.DB = cfg.RedisDB

	client := redis.NewClient(opts)

	// Verify connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		client.Close()
		return nil, err
	}

	// Set defaults for circuit breaker
	threshold := cfg.FailureThreshold
	if threshold <= 0 {
		threshold = 3
	}
	resetTimeout := cfg.ResetTimeout
	if resetTimeout <= 0 {
		resetTimeout = 30 * time.Second
	}

	return &RedisRateLimiter{
		client:       client,
		fallback:     NewRateLimiter(cfg.RequestsPerSecond, cfg.Burst),
		script:       redis.NewScript(tokenBucketScript),
		keyPrefix:    "ratelimit:",
		maxTokens:    cfg.Burst,
		refillRate:   cfg.RequestsPerSecond,
		threshold:    threshold,
		resetTimeout: resetTimeout,
	}, nil
}

// Allow checks if a request is allowed using the token bucket algorithm.
// Uses the default global key for rate limiting.
func (rl *RedisRateLimiter) Allow() bool {
	return rl.AllowKey("global")
}

// AllowKey checks if a request is allowed for a specific key (IP, API key, etc.).
func (rl *RedisRateLimiter) AllowKey(key string) bool {
	// Check circuit breaker state
	if rl.isCircuitOpen() {
		return rl.fallback.Allow()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	now := float64(time.Now().Unix())
	result, err := rl.script.Run(ctx, rl.client, []string{rl.keyPrefix + key},
		rl.maxTokens, rl.refillRate, now).Int()

	if err != nil {
		rl.recordFailure()
		log.Printf("[RateLimit] Redis error, using fallback: %v", err)
		return rl.fallback.Allow()
	}

	rl.recordSuccess()
	return result == 1
}

// isCircuitOpen checks if the circuit breaker is open.
func (rl *RedisRateLimiter) isCircuitOpen() bool {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	if !rl.circuitOpen {
		return false
	}

	// Check if it's time to try again
	if time.Since(rl.lastFailure) > rl.resetTimeout {
		return false // Allow a trial request
	}

	return true
}

// recordFailure records a Redis failure and potentially opens the circuit.
func (rl *RedisRateLimiter) recordFailure() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	rl.failures++
	rl.lastFailure = time.Now()

	if rl.failures >= rl.threshold {
		if !rl.circuitOpen {
			log.Printf("[RateLimit] Circuit breaker OPEN after %d failures", rl.failures)
		}
		rl.circuitOpen = true
	}
}

// recordSuccess records a successful Redis operation and resets the circuit.
func (rl *RedisRateLimiter) recordSuccess() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if rl.circuitOpen {
		log.Println("[RateLimit] Circuit breaker CLOSED, Redis recovered")
	}
	rl.failures = 0
	rl.circuitOpen = false
}

// Close closes the Redis connection.
func (rl *RedisRateLimiter) Close() error {
	if rl.client != nil {
		return rl.client.Close()
	}
	return nil
}

// HealthCheck verifies Redis connectivity.
func (rl *RedisRateLimiter) HealthCheck(ctx context.Context) error {
	return rl.client.Ping(ctx).Err()
}
