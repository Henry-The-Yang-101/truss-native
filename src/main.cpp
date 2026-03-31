#include "httplib.h"
#include "llama_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <mutex>

using json = nlohmann::json;

std::unique_ptr<LlamaEngine> engine = nullptr;

std::mutex engine_mutex;

std::string current_model_family = "";

std::string format_prompt(const std::string& raw_prompt, const std::string& family) {
    if (family == "llama3") {
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + raw_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (family == "qwen") {
        return "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }
    return raw_prompt;
}

int main() {
    httplib::Server svr;

    svr.Post("/v1/initialize", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);
        try {
            auto req_body = json::parse(req.body);
            std::string model_path = req_body["model_path"];
            current_model_family = req_body.value("family", "llama3"); // defaults to llama3

            std::cout << "\n[API] Unloading previous model from memory (if any)..." << std::endl;
            engine.reset(); // This automatically calls the LlamaEngine Destructor and frees the GPU memory

            std::cout << "[API] Loading new model: " << model_path << "..." << std::endl;
            engine = std::make_unique<LlamaEngine>(model_path);

            res.set_content(json({{"status", "success"}, {"message", "Model loaded into RAM"}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", std::string("Initialization failed: ") + e.what()}}).dump(), "application/json");
        }
    });

    // --- ENDPOINT 2: Generate text using loaded model ---
    svr.Post("/v1/predict", [&](const httplib::Request &req, httplib::Response &res) {
        std::lock_guard<std::mutex> lock(engine_mutex);

        // Safety check: Did they load a model yet?
        if (!engine) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"message", "No model is currently loaded in RAM. Call /v1/initialize first."}}).dump(), "application/json");
            return;
        }

        auto req_body = json::parse(req.body);
        std::string raw_prompt = req_body["prompt"];
        
        // Wrap the text in the correct model's syntax before sending to the GPU
        std::string formatted_prompt = format_prompt(raw_prompt, current_model_family);
        std::string output = engine->generate(formatted_prompt);

        json res_body = {{"model_output", output}};
        res.set_content(res_body.dump(), "application/json");
    });

    std::cout << "Dynamic API Server running on port 8080" << std::endl;
    std::cout << "Waiting for an initialization request..." << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}