#!/bin/bash
# Build script for LTEngine on macOS with Metal support

set -e  # Exit on error

echo "=== LTEngine macOS Build Script ==="
echo ""

# Add Homebrew to PATH
export PATH="/opt/homebrew/bin:$PATH"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed"
    echo "Install it from: https://brew.sh"
    exit 1
fi

# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    brew install cmake
fi

# Check if libomp is installed
if ! brew list libomp &> /dev/null; then
    echo "Installing libomp..."
    brew install libomp
fi

# Set OpenMP library paths for Cargo build
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Set Rust-specific environment variables
# This ensures the linker flags are passed through Cargo to the C/C++ compiler
export RUSTFLAGS="-C link-arg=-L/opt/homebrew/opt/libomp/lib -C link-arg=-lomp"

# Set deployment target to match system version (fixes version mismatch warning)
export MACOSX_DEPLOYMENT_TARGET=11.0

# Clean previous build
echo ""
echo "Cleaning previous build..."
cargo clean

# Build with Metal support
echo ""
echo "Building LTEngine with Metal support..."
echo "This may take several minutes..."
echo ""

cargo build --features metal --release

# Check if build succeeded
if [ -f "target/release/ltengine" ]; then
    echo ""
    echo "==================================="
    echo "Build completed successfully! 🎉"
    echo "==================================="
    echo ""
    echo "Binary location: target/release/ltengine"
    echo "Binary size: $(du -h target/release/ltengine | cut -f1)"
    echo ""
    echo "Test the binary with:"
    echo "  ./target/release/ltengine --help"
    echo ""
else
    echo ""
    echo "Build failed!"
    exit 1
fi
