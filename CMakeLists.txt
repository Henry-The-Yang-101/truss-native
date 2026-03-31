cmake_minimum_required(VERSION 3.24)
project(TrussLlamaServer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# Fetch llama.cpp directly from GitHub
include(FetchContent)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    GIT_TAG master
)
# Force Apple Silicon GPU acceleration
set(GGML_METAL ON CACHE BOOL "Enable Metal" FORCE) 
FetchContent_MakeAvailable(llama)

add_executable(truss_server src/main.cpp src/llama_engine.cpp)

target_link_libraries(truss_server PRIVATE 
    yaml-cpp
    nlohmann_json::nlohmann_json
    llama
)