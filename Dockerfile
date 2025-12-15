# ============================================
# DenseCore Production Dockerfile
# Multi-stage build with optimized layer caching
# ============================================
# 
# Cache Optimization Strategy:
# 1. Copy dependency files (ggml, CMake) first
# 2. Build ggml (slow, but rarely changes)
# 3. Copy application source
# 4. Build DenseCore (fast, changes frequently)
#
# This ensures changing a .cpp file doesn't rebuild ggml

# Alpine version pinned for reproducibility
ARG ALPINE_VERSION=3.21

# ============================================
# Stage 1: Dependency Builder (ggml cache layer)
# ============================================
FROM golang:1.24-alpine${ALPINE_VERSION} AS deps-builder

RUN apk add --no-cache \
    git \
    make \
    gcc \
    g++ \
    cmake \
    linux-headers

WORKDIR /app

# Copy ONLY dependency files first (ggml + CMake config)
# This layer will be cached unless these files change
COPY core/third_party/ core/third_party/
COPY core/CMakeLists.txt core/CMakeLists.txt
COPY core/include/ core/include/

# Create stub sources to satisfy CMake (will be replaced later)
RUN mkdir -p core/src core/src/quantization core/src/pruning core/tests && \
    for f in engine worker inference model_types model_loader tokenizer kv_cache \
    optimization_bridge save_model quantizer pruner quantize version tensor_utils; do \
    echo "" > core/src/${f}.cpp; \
    done && \
    for f in max_quantizer awq_quantizer smoothquant_quantizer; do \
    echo "" > core/src/quantization/${f}.cpp; \
    done && \
    for f in depth_pruner width_pruner attention_pruner combined_pruner; do \
    echo "" > core/src/pruning/${f}.cpp; \
    done && \
    for f in test_simd_ops test_memory_pool test_kv_cache; do \
    echo "" > core/tests/${f}.cpp; \
    done

# Pre-build ggml (this is the slow part - now cached)
RUN mkdir -p build && cd build && \
    cmake ../core -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target ggml -j$(nproc)

# ============================================
# Stage 2: Application Builder
# ============================================
FROM deps-builder AS builder

# Install Go protobuf plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

# Install protobuf
RUN apk add --no-cache protobuf protobuf-dev

# Copy Go module files for caching
COPY server/go.mod server/go.sum* ./server/
WORKDIR /app/server
RUN go mod download

# Now copy the REAL source files (invalidates from here down on code changes)
WORKDIR /app
COPY core/src/ core/src/
COPY server/ server/
COPY proto/ proto/
COPY Makefile ./

# Generate Proto files (optional, may fail if not needed)
RUN protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    proto/densecore.proto 2>/dev/null || true

# Rebuild only DenseCore (ggml is already built and cached)
RUN cd build && \
    cmake --build . --target densecore -j$(nproc)

# Build Go server
WORKDIR /app/server
ENV CGO_LDFLAGS="-L/app/build -ldensecore -lstdc++ -ldl"
ENV CGO_CFLAGS="-I/app/core/include"
RUN CGO_ENABLED=1 GOOS=linux go build -o /densecore-server .

# ============================================
# Stage 3: Runtime
# ============================================
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    libgcc \
    libgomp \
    ca-certificates \
    wget \
    tini

# Create non-root user
RUN addgroup -g 1000 densecore && \
    adduser -u 1000 -G densecore -D densecore

# Create directories
RUN mkdir -p /app/models /app/lib && \
    chown -R densecore:densecore /app

WORKDIR /app

# Copy binaries and libraries
COPY --from=builder /densecore-server /app/densecore-server
COPY --from=builder /app/build/libdensecore.so* /app/lib/
COPY --from=builder /app/build/libggml*.so* /app/lib/

# Create symlink for versioned library name
RUN cd /app/lib && \
    [ -f libdensecore.so.1 ] || ln -s libdensecore.so libdensecore.so.1

# Set library path
ENV LD_LIBRARY_PATH=/app/lib

# Switch to non-root user
USER densecore

# Default configuration
ENV PORT=8080 \
    HOST=0.0.0.0 \
    READ_TIMEOUT=30s \
    WRITE_TIMEOUT=120s \
    SHUTDOWN_TIMEOUT=30s \
    RATE_LIMIT_ENABLED=true \
    RATE_LIMIT_RPS=100 \
    LOG_FORMAT=json \
    THREADS=0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD wget -q --spider http://localhost:8080/health/live || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Run the server
CMD ["/app/densecore-server"]
