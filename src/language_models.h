#pragma once
#include "llm_engine.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <memory>
#include <string>

class LargeLanguageModel {
protected:
    std::unique_ptr<LLMEngine> engine_;

public:
    LargeLanguageModel(const std::string& model_path) {
        engine_ = std::make_unique<LLMEngine>(model_path);
    }
    virtual ~LargeLanguageModel() = default;
    virtual std::string format_messages(const nlohmann::json& messages) const = 0;

    std::string generate(const nlohmann::json& messages, const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr) {
        std::string formatted_prompt = format_messages(messages);
        return engine_->generate(formatted_prompt, config, std::move(token_callback));
    }
};

class LlamaLLM : public LargeLanguageModel {
public:
    LlamaLLM(const std::string& model_path) : LargeLanguageModel(model_path) {}

    std::string format_messages(const nlohmann::json& messages) const override {
        std::string out = "<|begin_of_text|>";
        for (const auto& msg : messages) {
            std::string role = msg.at("role").get<std::string>();
            std::string content = msg.at("content").get<std::string>();
            out += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }
};

class QwenLLM : public LargeLanguageModel {
public:
    QwenLLM(const std::string& model_path) : LargeLanguageModel(model_path) {}

    std::string format_messages(const nlohmann::json& messages) const override {
        std::string out;
        for (const auto& msg : messages) {
            std::string role = msg.at("role").get<std::string>();
            std::string content = msg.at("content").get<std::string>();
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        out += "<|im_start|>assistant\n";
        return out;
    }
};

class Llama3_8B : public LlamaLLM {
public:
    Llama3_8B() : LlamaLLM("../models/llama3-8b-gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf") {}
};

class Qwen2_5_32B : public QwenLLM {
public:
    // Qwen2_5_32B() : QwenLLM("../models/qwen2.5-32b-gguf/Qwen2.5-32B-Instruct-Q3_K_L.gguf") {} 
    Qwen2_5_32B() : QwenLLM("../models/qwen2.5-32b-gguf/qwen2.5-coder-32b-instruct-q4_k_m.gguf") {} 
};

class Qwen3Coder_30B : public QwenLLM {
public:
    Qwen3Coder_30B() : QwenLLM("../models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf") {}
};