#include "httplib.h"
#include "config_parser.h"
#include "llama_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    TrussConfig config = load_config("../config.yaml");

    std::string model_file = config.model_dir + "/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";

    std::cout << "Loading model from " << model_file << "..." << std::endl;
    LlamaEngine engine(model_file);

    httplib::Server svr;

    svr.Post("/v1/predict", [&](const httplib::Request &req, httplib::Response &res) {
        auto req_body = json::parse(req.body);
        std::string prompt = req_body["prompt"];

        std::string output = engine.generate(prompt);

        json res_body = {{"model_output", output}};
        res.set_content(res_body.dump(), "application/json");
    });

    std::cout << "Native Truss-Llama Server running on port " << config.port << std::endl;
    svr.listen("0.0.0.0", config.port);
    return 0;
}