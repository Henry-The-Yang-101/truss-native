#include "httplib.h"
#include "language_models.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>

using json = nlohmann::json;

std::unique_ptr<LargeLanguageModel> active_model = nullptr;
std::string active_model_id;
std::mutex engine_mutex;

static bool validate_chat_messages(const json& messages) {
    if (!messages.is_array() || messages.empty()) {
        return false;
    }
    for (const auto& msg : messages) {
        if (!msg.is_object() || !msg.contains("role") || !msg.contains("content")) {
            return false;
        }
        const auto& role = msg["role"];
        const auto& content = msg["content"];
        if (!role.is_string() || !content.is_string()) {
            return false;
        }
    }
    return true;
}

int main() {
    httplib::Server svr;

    svr.Post("/v1/initialize", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);
        try {
            auto req_body = json::parse(req.body);
            std::string keyword = req_body["model"];

            std::cout << "\n[API] Unloading previous model from memory..." << std::endl;
            active_model.reset();
            active_model_id.clear();

            std::cout << "[API] Initializing model keyword: " << keyword << "..." << std::endl;

            if (keyword == "llama-3") {
                active_model = std::make_unique<Llama3_8B>();
                active_model_id = "llama-3";
            } else if (keyword == "qwen-2.5") {
                active_model = std::make_unique<Qwen2_5_32B>();
                active_model_id = "qwen-2.5";
            } else if (keyword == "qwen-3-coder") {
                active_model = std::make_unique<Qwen3Coder_30B>();
                active_model_id = "qwen-3-coder";
            } else {
                throw std::runtime_error("Unknown model keyword. Available: llama-3, qwen-2.5, qwen-3-coder");
            }

            res.set_content(json({{"status", "success"}, {"message", keyword + " loaded into RAM"}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", std::string("Initialization failed: ") + e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);

        try {
            if (!active_model) {
                res.status = 400;
                res.set_content(
                    json({{"error", json{{"message", "No model loaded. Call /v1/initialize first."},
                                      {"type", "invalid_request_error"},
                                      {"param", nullptr},
                                      {"code", "model_not_found"}}}})
                        .dump(),
                    "application/json");
                return;
            }

            auto req_body = json::parse(req.body);

            if (req_body.contains("model") && req_body["model"].is_string()) {
                std::string requested_model = req_body["model"].get<std::string>();
                if (requested_model != active_model_id) {
                    res.status = 400;
                    res.set_content(
                        json({{"error", json{{"message", "Model mismatch. Currently initialized: " + active_model_id + ". Requested: " + requested_model + ". Call /v1/initialize to switch models."},
                                          {"type", "invalid_request_error"},
                                          {"param", "model"},
                                          {"code", "model_not_found"}}}})
                            .dump(),
                        "application/json");
                    return;
                }
            }

            if (!req_body.contains("messages") || !validate_chat_messages(req_body["messages"])) {
                res.status = 400;
                res.set_content(
                    json({{"error", json{{"message", "Invalid or missing 'messages' array (each item needs string 'role' and 'content')."},
                                      {"type", "invalid_request_error"},
                                      {"param", "messages"},
                                      {"code", nullptr}}}})
                        .dump(),
                    "application/json");
                return;
            }

            GenerationConfig config;
            if (req_body.contains("temperature") && req_body["temperature"].is_number()) {
                config.temperature = static_cast<float>(req_body["temperature"].get<double>());
            }
            if (req_body.contains("max_tokens") && req_body["max_tokens"].is_number()) {
                config.max_tokens = static_cast<int>(req_body["max_tokens"].get<double>());
            }

            const json& messages = req_body["messages"];
            std::string output = active_model->generate(messages, config);

            const auto created = std::chrono::duration_cast<std::chrono::seconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();

            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::string id = "chatcmpl-" + std::to_string(gen());

            json res_body = {
                {"id", id},
                {"object", "chat.completion"},
                {"created", created},
                {"model", active_model_id},
                {"choices",
                 json::array({
                     json{{"index", 0},
                          {"message", json{{"role", "assistant"}, {"content", output}}},
                          {"finish_reason", "stop"}},
                 })}};
            res.set_content(res_body.dump(), "application/json");
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", json{{"message", std::string("Malformed JSON body: ") + e.what()},
                                                  {"type", "invalid_request_error"},
                                                  {"param", nullptr},
                                                  {"code", nullptr}}}})
                                .dump(),
                            "application/json");
        }
    });

    svr.Get("/v1/models", [&](const httplib::Request &req, httplib::Response &res) {
        json models_list = {
            {"object", "list"},
            {"data",
             json::array({
                 {{"id", "llama-3"}, {"object", "model"}, {"created", 1715299200}, {"owned_by", "truss-native"}},
                 {{"id", "qwen-2.5"}, {"object", "model"}, {"created", 1715299200}, {"owned_by", "truss-native"}},
                 {{"id", "qwen-3-coder"}, {"object", "model"}, {"created", 1715299200}, {"owned_by", "truss-native"}},
             })}};

        res.set_content(models_list.dump(), "application/json");
    });

    std::cout << "Dynamic API Server running on port 8080" << std::endl;
    std::cout << "Waiting for an initialization request..." << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}