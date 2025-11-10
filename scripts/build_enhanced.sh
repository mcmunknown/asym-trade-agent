#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
CPP_DIR="${PROJECT_ROOT}/cpp"

echo "🚀 Building Enhanced MathCore with C++ Integration..."
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"
echo "C++ Source: ${CPP_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo "📦 Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
    -DPython3_EXECUTABLE="$(command -v python3)"

# Build
echo "🔨 Building..."
make -j$(nproc 2>/dev/null || echo 4)

# Test
echo "🧪 Running tests..."
if command -v ctest >/dev/null 2>&1; then
    ctest --output-on-failure
else
    echo "ctest not found, skipping tests"
fi

# Install
echo "📦 Installing..."
make install

# Test Python import
echo "🐍 Testing Python import..."
cd "${PROJECT_ROOT}"
export PYTHONPATH="${BUILD_DIR}/install/lib/python3/site-packages:${PYTHONPATH}"

if python3 -c "
import mathcore
print('✅ Enhanced mathcore module imported successfully')
print(f'📊 Version: {mathcore.cpp_version()}')
print(f'🔧 C++ Available: {mathcore.cpp_available()}')
"; then
    echo "✅ Build completed successfully!"
    echo "📍 Installation: ${BUILD_DIR}/install"
    echo "🐍 Test with: PYTHONPATH=\"${BUILD_DIR}/install/lib/python3/site-packages\" python3"
else
    echo "❌ Python import test failed"
    exit 1
fi

echo "🎉 Enhanced C++ build complete!"
echo ""
echo "📋 Usage Instructions:"
echo "1. Set PYTHONPATH: export PYTHONPATH=\"${BUILD_DIR}/install/lib/python3/site-packages:\$PYTHONPATH\""
echo "2. Import in Python: import mathcore"
echo "3. Use enhanced functions: analyzer = create_enhanced_analyzer()"
