#include "httplib.h"
#include "language_models.h"
#include "inference_queue.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <thread>
#include <atomic>
#include <vector>

using json = nlohmann::json;

static ModelRegistry g_registry("../models/models.yaml");

std::unique_ptr<LargeLanguageModel> active_model = nullptr;
std::string active_model_id;
std::mutex init_mutex;

InferenceQueue req_queue;
std::atomic<bool> server_running{true};

std::vector<json> global_chat_history;
size_t last_client_message_count = 0;
std::mutex history_mutex;

std::mutex completion_mutex;

static constexpr float SUMMARIZE_THRESHOLD = 0.75f;
static constexpr int   RECENT_MESSAGES_TO_KEEP = 4;

static bool validate_chat_messages(const json& messages) {
    if (!messages.is_array() || messages.empty()) {
        return false;
    }
    for (const auto& msg : messages) {
        if (!msg.is_object() || !msg.contains("role") || !msg.contains("content")) {
            return false;
        }
        if (!msg["role"].is_string() || !msg["content"].is_string()) {
            return false;
        }
    }
    return true;
}

static std::string run_inference(const json& messages, const GenerationConfig& config,
                                  std::function<bool(const std::string&)> token_cb = nullptr) {
    InferenceRequest req;
    req.messages_payload = messages;
    req.config = config;
    req.token_callback = std::move(token_cb);
    std::promise<std::string> promise;
    auto future = promise.get_future();
    req.promise = std::move(promise);
    req_queue.push_request(std::move(req));
    return future.get();
}

void inference_worker_loop() {
    while (server_running) {
        InferenceRequest req = req_queue.pop_request();

        if (!server_running) {
            try {
                req.promise.set_exception(std::make_exception_ptr(std::runtime_error("Server shutting down")));
            } catch (...) {}
            break;
        }

        try {
            LargeLanguageModel* current_model = nullptr;
            {
                std::lock_guard<std::mutex> lock(init_mutex);
                if (active_model) {
                    current_model = active_model.get();
                }
            }

            if (!current_model) {
                req.promise.set_exception(std::make_exception_ptr(std::runtime_error("Model was unloaded before request could be processed.")));
                continue;
            }

            std::string result = current_model->generate(req.messages_payload, req.config, req.token_callback);
            req.promise.set_value(result);
        } catch (...) {
            try {
                req.promise.set_exception(std::current_exception());
            } catch (...) {}
        }
    }
}

static bool maybe_summarize(LargeLanguageModel* model) {
    int max_ctx = model->get_max_context();
    if (max_ctx <= 0) return false;

    json history_snapshot;
    int first_summarizable;
    int last_summarizable;
    {
        std::lock_guard<std::mutex> hlock(history_mutex);

        std::string formatted = model->format_messages(json(global_chat_history));
        int token_count = model->count_tokens(formatted);
        float usage = static_cast<float>(token_count) / static_cast<float>(max_ctx);
        if (usage < SUMMARIZE_THRESHOLD) return false;

        first_summarizable = 0;
        if (!global_chat_history.empty() && global_chat_history[0]["role"] == "system") {
            first_summarizable = 1;
        }
        last_summarizable = (int)global_chat_history.size() - RECENT_MESSAGES_TO_KEEP;
        if (last_summarizable <= first_summarizable) return false;

        history_snapshot = json(global_chat_history);
    }

    std::string conv_text;
    for (int i = first_summarizable; i < last_summarizable; i++) {
        conv_text += history_snapshot[i]["role"].get<std::string>() + ": "
                   + history_snapshot[i]["content"].get<std::string>() + "\n";
    }

    json summarize_messages = json::array({
        json{{"role", "user"},
             {"content", "Summarize the following conversation concisely, preserving all important facts, decisions, and context:\n\n" + conv_text}}
    });

    GenerationConfig sum_config;
    sum_config.max_tokens = 512;
    sum_config.temperature = 0.3f;

    std::cout << "[Summarization] Condensing " << (last_summarizable - first_summarizable)
              << " messages..." << std::endl;

    std::string summary = run_inference(summarize_messages, sum_config);

    {
        std::lock_guard<std::mutex> hlock(history_mutex);

        std::vector<json> new_history;
        if (first_summarizable > 0) {
            new_history.push_back(global_chat_history[0]);
        }
        new_history.push_back(json{{"role", "system"},
                                   {"content", "Summary of previous conversation: " + summary}});
        for (int i = last_summarizable; i < (int)global_chat_history.size(); i++) {
            new_history.push_back(global_chat_history[i]);
        }
        global_chat_history = std::move(new_history);
    }

    std::cout << "[Summarization] Done. History condensed to "
              << global_chat_history.size() << " messages." << std::endl;
    return true;
}

int main() {
    httplib::Server svr;

    std::thread inference_worker(inference_worker_loop);

    svr.Post("/v1/initialize", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(init_mutex);
        try {
            auto req_body = json::parse(req.body);
            std::string model_id = req_body["model"];
            int max_context = 0;
            if (req_body.contains("max_context") && req_body["max_context"].is_number()) {
                max_context = req_body["max_context"].get<int>();
            }

            std::cout << "\n[API] Unloading previous model from memory..." << std::endl;
            active_model.reset();
            active_model_id.clear();

            {
                std::lock_guard<std::mutex> hlock(history_mutex);
                global_chat_history.clear();
                last_client_message_count = 0;
            }

            std::cout << "[API] Initializing model: " << model_id << "..." << std::endl;

            active_model    = g_registry.load(model_id, max_context);
            active_model_id = model_id;

            res.set_content(json({{"status", "success"},
                                  {"message", model_id + " loaded into RAM"}})
                                .dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", std::string("Initialization failed: ") + e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/context_reset", [&](const httplib::Request &req, httplib::Response &res) {
        {
            std::lock_guard<std::mutex> hlock(history_mutex);
            global_chat_history.clear();
            last_client_message_count = 0;
        }
        {
            std::lock_guard<std::mutex> lock(init_mutex);
            if (active_model) active_model->reset_cache();
        }
        res.set_content(json({{"status", "success"}, {"message", "Context cleared."}}).dump(), "application/json");
    });

    svr.Get("/v1/context_usage", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(init_mutex);
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

        int max_ctx = active_model->get_max_context();
        int used_ctx = 0;

        {
            std::lock_guard<std::mutex> hlock(history_mutex);
            std::string formatted = active_model->format_messages(json(global_chat_history));
            used_ctx = active_model->count_tokens(formatted);
        }

        json res_body = {
            {"used_context", used_ctx},
            {"max_context", max_ctx}
        };
        res.set_content(res_body.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request &req, httplib::Response &res) {
        json req_body;

        try {
            req_body = json::parse(req.body);
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", json{{"message", std::string("Malformed JSON body: ") + e.what()},
                                                  {"type", "invalid_request_error"},
                                                  {"param", nullptr},
                                                  {"code", nullptr}}}})
                                .dump(),
                            "application/json");
            return;
        }

        const bool stream_requested =
            req_body.contains("stream") && req_body["stream"].is_boolean() && req_body["stream"].get<bool>();

        GenerationConfig config;
        if (req_body.contains("temperature") && req_body["temperature"].is_number()) {
            config.temperature = static_cast<float>(req_body["temperature"].get<double>());
        }
        if (req_body.contains("max_tokens") && req_body["max_tokens"].is_number()) {
            config.max_tokens = static_cast<int>(req_body["max_tokens"].get<double>());
        }

        {
            std::lock_guard<std::mutex> lock(init_mutex);
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

        json client_messages = req_body["messages"];

        std::lock_guard<std::mutex> comp_lock(completion_mutex);

        {
            std::lock_guard<std::mutex> hlock(history_mutex);
            size_t client_count = client_messages.size();
            if (client_count < last_client_message_count) {
                global_chat_history.clear();
                last_client_message_count = 0;
                std::lock_guard<std::mutex> mlock(init_mutex);
                if (active_model) active_model->reset_cache();
            }
            for (size_t i = last_client_message_count; i < client_count; i++) {
                global_chat_history.push_back(client_messages[i]);
            }
            last_client_message_count = client_count;
        }

        LargeLanguageModel* model_ptr = nullptr;
        {
            std::lock_guard<std::mutex> mlock(init_mutex);
            model_ptr = active_model.get();
        }

        bool summarization_triggered = maybe_summarize(model_ptr);

        json messages_for_generation;
        {
            std::lock_guard<std::mutex> hlock(history_mutex);
            messages_for_generation = json(global_chat_history);
        }

        if (stream_requested) {
            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::string completion_id = "chatcmpl-" + std::to_string(gen());

            res.set_chunked_content_provider(
                "text/event-stream",
                [completion_id = std::move(completion_id),
                 messages_for_generation,
                 config,
                 summarization_triggered](size_t offset, httplib::DataSink &sink) mutable -> bool {
                    if (offset > 0) {
                        sink.done();
                        return true;
                    }

                    std::string assistant_response;

                    auto token_cb = [&completion_id, &sink, &assistant_response](const std::string &piece) -> bool {
                        assistant_response += piece;
                        json choice = json::object({{"index", 0},
                                                    {"delta", json::object({{"content", piece}})},
                                                    {"finish_reason", nullptr}});
                        json chunk = json::object({{"id", completion_id},
                                                   {"object", "chat.completion.chunk"},
                                                   {"choices", json::array({choice})}});
                        std::string line = std::string("data: ") + chunk.dump() + "\n\n";
                        if (!sink.write(line.data(), line.size())) {
                            return false;
                        }
                        return sink.is_writable();
                    };

                    try {
                        run_inference(messages_for_generation, config, token_cb);
                    } catch (...) {}

                    {
                        std::lock_guard<std::mutex> hlock(history_mutex);
                        if (!assistant_response.empty()) {
                            global_chat_history.push_back(json{{"role", "assistant"}, {"content", assistant_response}});
                            last_client_message_count = global_chat_history.size();
                        }
                    }

                    json final_chunk = json::object({
                        {"id", completion_id},
                        {"object", "chat.completion.chunk"},
                        {"summarization_triggered", summarization_triggered},
                        {"choices", json::array({json::object({
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", "stop"}
                        })})}
                    });
                    std::string final_line = std::string("data: ") + final_chunk.dump() + "\n\n";
                    sink.write(final_line.data(), final_line.size());

                    constexpr const char k_done[] = "data: [DONE]\n\n";
                    sink.write(k_done, sizeof(k_done) - 1);
                    sink.done();
                    return true;
                });
            return;
        }

        std::string output;
        try {
            output = run_inference(messages_for_generation, config);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", json{{"message", e.what()},
                                              {"type", "internal_server_error"},
                                              {"param", nullptr},
                                              {"code", nullptr}}}})
                            .dump(), "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> hlock(history_mutex);
            global_chat_history.push_back(json{{"role", "assistant"}, {"content", output}});
            last_client_message_count = global_chat_history.size();
        }

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
            {"summarization_triggered", summarization_triggered},
            {"choices",
             json::array({
                 json{{"index", 0},
                      {"message", json{{"role", "assistant"}, {"content", output}}},
                      {"finish_reason", "stop"}},
             })}};
        res.set_content(res_body.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &req, httplib::Response &res) {
        json data = json::array();
        for (const auto& spec : g_registry.all()) {
            data.push_back({
                {"id",       spec.id},
                {"object",   "model"},
                {"created",  1715299200},
                {"owned_by", "truss-native"},
                {"type",     spec.type}
            });
        }
        json models_list = {{"object", "list"}, {"data", data}};
        res.set_content(models_list.dump(), "application/json");
    });

    std::cout << "Truss Native Started!" << std::endl;
    std::cout << "Dynamic API Server running on port 8080" << std::endl;
    std::cout << "Waiting for an initialization request..." << std::endl;
    svr.listen("0.0.0.0", 8080);

    server_running = false;

    InferenceRequest dummy_req;
    std::promise<std::string> dummy_promise;
    dummy_req.promise = std::move(dummy_promise);
    req_queue.push_request(std::move(dummy_req));

    inference_worker.join();

    return 0;
}
