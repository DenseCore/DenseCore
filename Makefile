.PHONY: all clean lib server run test help docs

# Default target
all: lib server

# Directories
BUILD_DIR := build
CORE_DIR := core
SERVER_DIR := server
BIN_DIR := bin

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
    LIB_PREFIX := lib
else ifeq ($(UNAME_S),Linux)
    LIB_EXT := so
    LIB_PREFIX := lib
else
    LIB_EXT := dll
    LIB_PREFIX :=
endif

LIB_NAME := $(LIB_PREFIX)densecore.$(LIB_EXT)

help:
	@echo "=========================================="
	@echo "DenseCore Build System"
	@echo "=========================================="
	@echo "Available targets:"
	@echo "  make all     - Build library and server (default)"
	@echo "  make lib     - Build C++ core library"
	@echo "  make server  - Build Go server"
	@echo "  make run     - Build and run the server"
	@echo "  make test    - Run a test request"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Show this help message"
	@echo "=========================================="
	@echo "Environment variables:"
	@echo "  MAIN_MODEL_PATH  - Path to main model"
	@echo "  DRAFT_MODEL_PATH - Path to draft model (optional)"
	@echo "  PORT             - Server port (default: 8080)"
	@echo "=========================================="

# Build C++ shared library
lib:
	@echo "=========================================="
	@echo "Building DenseCore C++ Library"
	@echo "=========================================="
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && $(MAKE)
	@echo ""
	@echo "Library built successfully: $(BUILD_DIR)/$(LIB_NAME)"
	@echo "=========================================="

# Build quantize tool
quantize:
	@echo "=========================================="
	@echo "Building Quantize Tool"
	@echo "=========================================="
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && $(MAKE) quantize
	@echo ""
	@echo "Quantize tool built successfully: $(BIN_DIR)/quantize"
	@echo "=========================================="

# Build Go server
server: lib
	@echo "=========================================="
	@echo "Building DenseCore Go Server"
	@echo "=========================================="
	@mkdir -p $(BIN_DIR)
	@cd $(SERVER_DIR) && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="-L../$(BUILD_DIR) -ldensecore -lstdc++" \
		go build -o ../$(BIN_DIR)/densecore-server .
	@echo ""
	@echo "Server built successfully: $(BIN_DIR)/densecore-server"
	@echo "=========================================="

# Build and run the server
run: server
	@echo "=========================================="
	@echo "Starting DenseCore Server"
	@echo "=========================================="
ifeq ($(UNAME_S),Darwin)
	@DYLD_LIBRARY_PATH=$(BUILD_DIR):$$DYLD_LIBRARY_PATH $(BIN_DIR)/densecore-server
else
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BIN_DIR)/densecore-server
endif

# Test the server with a sample request
test:
	@echo "=========================================="
	@echo "Testing DenseCore Server"
	@echo "=========================================="
	@echo "Sending test request to http://localhost:8080/v1/chat/completions"
	@echo ""
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "densecore-v1", \
			"messages": [{"role": "user", "content": "Hello, how are you?"}], \
			"max_tokens": 100 \
		}' | python3 -m json.tool || echo "Note: Install python3 for pretty JSON output"
	@echo ""
	@echo "=========================================="

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@rm -f $(SERVER_DIR)/densecore-server
	@echo "Clean complete"

# Install dependencies (for development)
deps:
	@echo "=========================================="
	@echo "Checking Dependencies"
	@echo "=========================================="
	@command -v cmake >/dev/null 2>&1 || { echo "❌ CMake not found. Please install CMake 3.14+"; exit 1; }
	@command -v go >/dev/null 2>&1 || { echo "❌ Go not found. Please install Go 1.19+"; exit 1; }
	@command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo "❌ C++ compiler not found"; exit 1; }
	@echo "✓ CMake found: $$(cmake --version | head -n1)"
	@echo "✓ Go found: $$(go version)"
	@echo "✓ C++ compiler found"
	@echo "=========================================="
	@echo "All dependencies satisfied!"
	@echo "=========================================="

# Build in debug mode
debug:
	@echo "Building in debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ../$(CORE_DIR) -DCMAKE_BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && $(MAKE)
	@cd $(SERVER_DIR) && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="-L../$(BUILD_DIR) -ldensecore -lstdc++" \
		go build -gcflags="all=-N -l" -o ../$(BIN_DIR)/densecore-server .
	@echo "Debug build complete"

# Generate API documentation
docs:
	@echo "=========================================="
	@echo "Generating API Documentation"
	@echo "=========================================="
	@mkdir -p docs/api
	@doxygen Doxyfile
	@echo ""
	@echo "Documentation generated at: docs/api/html/index.html"
	@echo "=========================================="
