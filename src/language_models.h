#pragma once
#include "llm_engine.h"
#include <string>
#include <memory>

class LargeLanguageModel {
protected:
    std::unique_ptr<LLMEngine> engine_;

public:
    LargeLanguageModel(const std::string& model_path) {
        engine_ = std::make_unique<LLMEngine>(model_path);
    }
    virtual ~LargeLanguageModel() = default;
    virtual std::string format_prompt(const std::string& raw_prompt) const = 0;

    std::string generate(const std::string& raw_prompt, const GenerationConfig& config = GenerationConfig()) {
        std::string formatted_prompt = format_prompt(raw_prompt);
        return engine_->generate(formatted_prompt, config);
    }
};

class LlamaLLM : public LargeLanguageModel {
public:
    LlamaLLM(const std::string& model_path) : LargeLanguageModel(model_path) {}
    
    std::string format_prompt(const std::string& raw_prompt) const override {
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + raw_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }
};

class QwenLLM : public LargeLanguageModel {
public:
    QwenLLM(const std::string& model_path) : LargeLanguageModel(model_path) {}
    
    std::string format_prompt(const std::string& raw_prompt) const override {
        return "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }
};

class Llama3_8B : public LlamaLLM {
public:
    Llama3_8B() : LlamaLLM("../models/llama3-8b-gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf") {}
};

class Qwen2_5_32B : public QwenLLM {
public:
    Qwen2_5_32B() : QwenLLM("../models/qwen2.5-32b-gguf/Qwen2.5-32B-Instruct-Q3_K_L.gguf") {} 
};

class Qwen3Coder_30B : public QwenLLM {
public:
    Qwen3Coder_30B() : QwenLLM("../models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf") {}
};