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
    explicit LlamaEngine(const std::string& model_path, bool flash_attention = true, int max_context = 0);
    ~LlamaEngine();

    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;

    std::string generate(const std::string& prompt, const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr);

    int count_tokens(const std::string& text) const;
    int get_max_context() const;
    void reset_cache();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};
