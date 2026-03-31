#pragma once
#include <string>
#include <memory>

struct GenerationConfig {
    int max_tokens = 100;
    float temperature = 0.7f;
};

class LLMEngine {
public:
    explicit LLMEngine(const std::string& model_path);
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    std::string generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig());

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};