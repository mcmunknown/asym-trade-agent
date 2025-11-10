#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "🚀 Building Simple MathCore with pybind11..."
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Directory: ${BUILD_DIR}"

# Clean previous build
if [ -d "${BUILD_DIR}" ]; then
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "📦 Configuring with CMake..."
# Use simple CMake that will work
cmake ../CMakeLists_simple.txt \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(which python3)"

echo "🔨 Building..."
# Build with minimal parallelization
make -j2

echo "✅ Build completed!"
echo "📍 Library location: ${BUILD_DIR}/mathcore$(python3 -c 'import sys; print(\".so\" if sys.platform != \"darwin\" else \".dylib\")')"

# Test import
echo "🧪 Testing import..."
cd "${PROJECT_ROOT}"
export PYTHONPATH="${BUILD_DIR}:$PYTHONPATH"

if python3 -c "import mathcore; print('✅ Import successful!'); print(f'Version: {mathcore.version()}'); print(f'C++ Available: {mathcore.cpp_available()}')"; then
    echo "🎉 MathCore built successfully!"
    echo ""
    echo "📋 Usage:"
    echo "   export PYTHONPATH=\"${BUILD_DIR}:\$PYTHONPATH\""
    echo "   python3 -c 'import mathcore; print(mathcore.version())'"
    echo ""
    echo "🚀 Test enhanced trading:"
    echo "   python3 -c \"from quantitative_models_enhanced import create_enhanced_analyzer; analyzer = create_enhanced_analyzer(prefer_cpp=True); print('Enhanced backend ready!' if hasattr(analyzer, 'use_cpp_backend') else 'Using fallback')\""
else
    echo "❌ Import failed - build may have issues"
    exit 1
fi
