# C++ Math Core Library CMake Configuration

set(MATHCORE_SOURCES
    math_core.c
    kalman_filter.cpp
    risk_kernels.cpp
    portfolio_opt.cpp
    websocket_client.cpp
)

set(MATHCORE_HEADERS
    math_core.h
    kalman_filter.h
    risk_kernels.h
    portfolio_opt.h
    websocket_client.h
)

# Create static library
add_library(mathcore_static STATIC ${MATHCORE_SOURCES} ${MATHCORE_HEADERS})

# Target-specific compile definitions
target_compile_definitions(mathcore_static PRIVATE MATHCORE_BUILD)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(mathcore_static PRIVATE ws2_32 wsock32)
else()
    target_link_libraries(mathcore_static PRIVATE ${CMAKE_DL_LIBS})
endif()

# Install headers and library
install(FILES ${MATHCORE_HEADERS} DESTINATION include/mathcore)
install(TARGETS mathcore_static DESTINATION lib)
