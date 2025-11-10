cmake_minimum_required(VERSION 3.16)
project(mathcore VERSION 2.0.0 LANGUAGES CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -Wextra")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(pybind11 REQUIRED)

# Python integration
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Python3_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(cpp)

# Math core library
add_library(mathcore_static STATIC
    cpp/math_core.cpp
    cpp/kalman_filter.cpp
    cpp/risk_kernels.cpp
    cpp/portfolio_opt.cpp
    cpp/websocket_client.cpp
)

target_include_directories(mathcore_static PRIVATE cpp)
target_link_libraries(mathcore_static PRIVATE ${CMAKE_DL_LIBS})

if(WIN32)
    target_link_libraries(mathcore_static PRIVATE ws2_32 wsock32)
endif()

# Python extension module
pybind11_add_module(mathcore SHARED
    cpp/bindings.cpp
)

target_link_libraries(mathcore PRIVATE mathcore_static)

# WebSocket client library (standalone)
add_library(websocket_client SHARED
    cpp/websocket_client.cpp
)

target_link_libraries(websocket_client PRIVATE 
    Boost::system
    Boost::asio
    ${CMAKE_DL_LIBS}
)

# Install targets
install(TARGETS mathcore websocket_client
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# Python package generation
set(PYTHON_MODULE_PATH ${Python3_SITEPACKAGES}/mathcore)
install(TARGETS mathcore DESTINATION ${PYTHON_MODULE_PATH})

# Testing
enable_testing()
add_subdirectory(tests)

# Configuration
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
    @ONLY
)

# Package generation
include(GNUInstallDirs)
set(CPACK_PACKAGE_NAME "mathcore")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION "High-performance mathematical kernels for calculus-based trading")
set(CPACK_PACKAGE_VENDOR "Anne's Trading System")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
else()
    set(CPACK_GENERATOR "DEB")
endif()

include(CPack)
