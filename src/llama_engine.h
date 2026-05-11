#pragma once
#include <functional>
#include <memory>
#include <string>

struct GenerationConfig {
    int max_tokens = 8192;
    float temperature = 0.7f;
};

class LlamaEngine {
public:
    explicit LlamaEngine(const std::string& model_path, bool flash_attention = true);
    ~LlamaEngine();

    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;

    std::string generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
