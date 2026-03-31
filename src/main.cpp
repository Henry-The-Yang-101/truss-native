#include "httplib.h"
#include "language_models.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <mutex>

using json = nlohmann::json;

std::unique_ptr<LargeLanguageModel> active_model = nullptr;
std::mutex engine_mutex;

int main() {
    httplib::Server svr;

    svr.Post("/v1/initialize", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);
        try {
            auto req_body = json::parse(req.body);
            std::string keyword = req_body["model"];

            std::cout << "\n[API] Unloading previous model from memory..." << std::endl;
            active_model.reset(); 

            std::cout << "[API] Initializing model keyword: " << keyword << "..." << std::endl;

            if (keyword == "llama-3") {
                active_model = std::make_unique<Llama3_8B>();
            } else if (keyword == "qwen-2.5") {
                active_model = std::make_unique<Qwen2_5_32B>();
            } else if (keyword == "qwen-3-coder") {
                active_model = std::make_unique<Qwen3Coder_30B>();
            } else {
                throw std::runtime_error("Unknown model keyword. Available: llama-3, qwen-2.5, qwen-3-coder");
            }

            res.set_content(json({{"status", "success"}, {"message", keyword + " loaded into RAM"}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", std::string("Initialization failed: ") + e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/v1/predict", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);

        if (!active_model) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", "No model loaded. Call /v1/initialize first."}}).dump(), "application/json");
            return;
        }

        auto req_body = json::parse(req.body);
        std::string raw_prompt = req_body["prompt"];
        
        std::string output = active_model->generate(raw_prompt);

        json res_body = {{"model_output", output}};
        res.set_content(res_body.dump(), "application/json");
    });

    std::cout << "Dynamic API Server running on port 8080" << std::endl;
    std::cout << "Waiting for an initialization request..." << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}