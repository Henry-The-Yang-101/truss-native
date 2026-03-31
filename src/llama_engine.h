#pragma once
#include <string>
#include <memory>

// FIX: Moved the struct outside the class so the compiler parses it first
struct GenerationConfig {
    int max_tokens = 100;
    float temperature = 0.7f;
};

class LlamaEngine {
public:
    explicit LlamaEngine(const std::string& model_path);
    ~LlamaEngine();

    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;

    std::string generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig());

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};