#pragma once
#include "llama_engine.h"
#include "mlx_engine.h"
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model metadata loaded from models.yaml
// ---------------------------------------------------------------------------

struct ModelSpec {
    std::string id;
    std::string type;         // "llama" | "mlx"
    std::string chat_format;  // "llama3" | "qwen"
    std::string path;
    bool flash_attention = true;
};

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------

class LargeLanguageModel {
public:
    virtual ~LargeLanguageModel() = default;

    virtual std::string format_messages(const nlohmann::json& messages) const = 0;

    virtual std::string generate(const nlohmann::json& messages,
                                 const GenerationConfig& config = GenerationConfig(),
                                 std::function<bool(const std::string&)> token_callback = nullptr) = 0;
};

// ---------------------------------------------------------------------------
// Llama-family (GGUF via llama.cpp)
// ---------------------------------------------------------------------------

class LlamaLLM : public LargeLanguageModel {
public:
    explicit LlamaLLM(const ModelSpec& spec)
        : engine_(spec.path, spec.flash_attention) {}

    std::string format_messages(const nlohmann::json& messages) const override {
        std::string out = "<|begin_of_text|>";
        for (const auto& msg : messages) {
            std::string role    = msg.at("role").get<std::string>();
            std::string content = msg.at("content").get<std::string>();
            out += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    std::string generate(const nlohmann::json& messages,
                         const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr) override {
        return engine_.generate(format_messages(messages), config, std::move(token_callback));
    }

private:
    LlamaEngine engine_;
};

// ---------------------------------------------------------------------------
// Qwen-family — inherits LlamaLLM, overrides only format_messages
// ---------------------------------------------------------------------------

class QwenLLM : public LlamaLLM {
public:
    explicit QwenLLM(const ModelSpec& spec) : LlamaLLM(spec) {}

    std::string format_messages(const nlohmann::json& messages) const override {
        std::string out;
        for (const auto& msg : messages) {
            std::string role    = msg.at("role").get<std::string>();
            std::string content = msg.at("content").get<std::string>();
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        out += "<|im_start|>assistant\n";
        return out;
    }
};

// ---------------------------------------------------------------------------
// MLX-family (safetensors via pybind11 + mlx_lm)
// ---------------------------------------------------------------------------

class MLXLLM : public LargeLanguageModel {
public:
    explicit MLXLLM(const ModelSpec& spec)
        : engine_(spec.path), chat_format_(spec.chat_format) {}

    std::string format_messages(const nlohmann::json& messages) const override {
        if (chat_format_ == "qwen") {
            std::string out;
            for (const auto& msg : messages) {
                std::string role    = msg.at("role").get<std::string>();
                std::string content = msg.at("content").get<std::string>();
                out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            }
            out += "<|im_start|>assistant\n";
            return out;
        }
        // Default: llama3 format
        std::string out = "<|begin_of_text|>";
        for (const auto& msg : messages) {
            std::string role    = msg.at("role").get<std::string>();
            std::string content = msg.at("content").get<std::string>();
            out += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + "<|eot_id|>";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    std::string generate(const nlohmann::json& messages,
                         const GenerationConfig& config = GenerationConfig(),
                         std::function<bool(const std::string&)> token_callback = nullptr) override {
        return engine_.generate(format_messages(messages), config, std::move(token_callback));
    }

private:
    MLXEngine engine_;
    std::string chat_format_;
};

// ---------------------------------------------------------------------------
// Model registry — reads models.yaml, creates LLM instances on demand
// ---------------------------------------------------------------------------

class ModelRegistry {
public:
    explicit ModelRegistry(const std::string& yaml_path) {
        YAML::Node root = YAML::LoadFile(yaml_path);
        for (const auto& node : root["models"]) {
            ModelSpec spec;
            spec.id           = node["id"].as<std::string>();
            spec.type         = node["type"].as<std::string>();
            spec.chat_format  = node["chat_format"].as<std::string>();
            spec.path         = node["path"].as<std::string>();
            spec.flash_attention = node["flash_attention"]
                                       ? node["flash_attention"].as<bool>()
                                       : true;
            specs_.push_back(std::move(spec));
        }
    }

    // Instantiate and return the model for the given id.
    std::unique_ptr<LargeLanguageModel> load(const std::string& id) const {
        const ModelSpec* spec = find(id);
        if (!spec) {
            throw std::runtime_error("Unknown model id '" + id + "'. Check models/models.yaml.");
        }

        if (spec->type == "llama") {
            if (spec->chat_format == "qwen") {
                return std::make_unique<QwenLLM>(*spec);
            }
            return std::make_unique<LlamaLLM>(*spec);
        }

        if (spec->type == "mlx") {
            return std::make_unique<MLXLLM>(*spec);
        }

        throw std::runtime_error("Unknown model type '" + spec->type + "' for id '" + id + "'.");
    }

    const std::vector<ModelSpec>& all() const { return specs_; }

private:
    std::vector<ModelSpec> specs_;

    const ModelSpec* find(const std::string& id) const {
        for (const auto& s : specs_) {
            if (s.id == id) return &s;
        }
        return nullptr;
    }
};
