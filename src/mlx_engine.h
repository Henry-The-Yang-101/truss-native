#pragma once
#include "llama_engine.h"  // for GenerationConfig
#include <functional>
#include <memory>
#include <string>

class MLXEngine {
public:
    // model_dir: path to the MLX model directory containing safetensors weights
    // and tokenizer files (tokenizer.json, tokenizer_config.json).
    explicit MLXEngine(const std::string& model_dir);
    ~MLXEngine();

    MLXEngine(const MLXEngine&) = delete;
    MLXEngine& operator=(const MLXEngine&) = delete;

    std::string generate(const std::string& prompt,
                         const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
