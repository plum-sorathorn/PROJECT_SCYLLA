cmake_minimum_required(VERSION 3.20)
project(ScyllaCppCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)

# ---- Vendored third-party libraries ----
set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${THIRD_PARTY}
    ${THIRD_PARTY}/crow/include
    ${THIRD_PARTY}/asio/include
    ${THIRD_PARTY}/lightgbm/include
    ${THIRD_PARTY}/curl/include
)

add_compile_definitions(ASIO_STANDALONE)

# Source files
set(SOURCES
    src/main.cpp
    src/data_fetcher.cpp
    src/metrics_engine.cpp
    src/api_handlers.cpp
    src/inference_engine.cpp
)

add_executable(scylla_core ${SOURCES})
target_compile_definitions(scylla_core PRIVATE ASIO_STANDALONE)

# ---- LightGBM C API (vendored DLL + import lib) ----
# Vendored by scripts/fetch_vendors.ps1 into third_party/lightgbm/.
# DLL goes in bin/ (runtime), import lib in lib/ (link time).
add_library(lightgbm SHARED IMPORTED)
set_target_properties(lightgbm PROPERTIES
    IMPORTED_LOCATION ${THIRD_PARTY}/lightgbm/bin/lib_lightgbm.dll
    IMPORTED_IMPLIB   ${THIRD_PARTY}/lightgbm/lib/lib_lightgbm.lib
    INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY}/lightgbm/include
)

# ---- libcurl (vendored DLL + generated import lib) ----
# Vendored by scripts/fetch_vendors.ps1 into third_party/curl/.
# Import lib is generated from the DLL's PE export table (see generate_libcurl_def.ps1).
add_library(curl SHARED IMPORTED)
set_target_properties(curl PROPERTIES
    IMPORTED_LOCATION ${THIRD_PARTY}/curl/bin/libcurl-x64.dll
    IMPORTED_IMPLIB   ${THIRD_PARTY}/curl/lib/libcurl.lib
    INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY}/curl/include
)

target_link_libraries(scylla_core
    Threads::Threads
    ws2_32           # Windows socket (used by libcurl and WinHTTP)
    wsock32
    lightgbm
    curl
)

# Optimization flags
if(MSVC)
    target_compile_options(scylla_core PRIVATE /O2 /EHsc /GL)
    target_link_options(scylla_core PRIVATE /LTCG)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|EM64T|x86_64")
        target_compile_options(scylla_core PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(scylla_core PRIVATE -O3 -Wall)
endif()

# ---- Post-build: stage runtime DLLs next to scylla_core.exe ----
# CMake's IMPORTED_LOCATION of a SHARED IMPORT lib is only used at link time;
# the DLL still has to land in the output directory (or on PATH) at runtime.
add_custom_command(TARGET scylla_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${THIRD_PARTY}/lightgbm/bin/lib_lightgbm.dll
        ${THIRD_PARTY}/curl/bin/libcurl-x64.dll
        $<TARGET_FILE_DIR:scylla_core>
    COMMENT "Staging LightGBM + libcurl DLLs next to scylla_core.exe"
)

message(STATUS "SCYLLA C++ Core configured. Build with: cmake --build . --config Release")
