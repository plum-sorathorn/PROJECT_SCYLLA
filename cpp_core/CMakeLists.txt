cmake_minimum_required(VERSION 3.20)
project(ScyllaCppCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)

# ---- Crow (header-only, vendored) ----
# Crow requires Boost.Asio or standalone Asio
# We use standalone Asio (also header-only, vendored in third_party/)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/crow/include
    ${CMAKE_SOURCE_DIR}/third_party/asio/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann
)

add_definitions(-DASIO_STANDALONE -DCROW_USE_BOOST=0)

# Source files
set(SOURCES
    src/main.cpp
    src/data_fetcher.cpp
    src/metrics_engine.cpp
    src/api_handlers.cpp
)

add_executable(scylla_core ${SOURCES})

target_link_libraries(scylla_core
    Threads::Threads
    ws2_32   # Windows socket
    wsock32
)

# Optimization flags
if(MSVC)
    target_compile_options(scylla_core PRIVATE /O2 /EHsc)
else()
    target_compile_options(scylla_core PRIVATE -O3 -Wall)
endif()

message(STATUS "SCYLLA C++ Core configured. Build with: cmake --build . --config Release")
