#include "mlx_engine.h"
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Python interpreter lifetime
//
// Only one scoped_interpreter may be alive at a time per process. We keep a
// reference-counted wrapper so that any number of MLXEngine instances can
// share the same interpreter without tearing it down prematurely.
// ---------------------------------------------------------------------------

namespace {

struct InterpreterGuard {
    py::scoped_interpreter interp;
};

static std::weak_ptr<InterpreterGuard> g_interp_weak;
static std::mutex                       g_interp_mutex;

std::shared_ptr<InterpreterGuard> acquire_interpreter() {
    std::lock_guard<std::mutex> lock(g_interp_mutex);
    auto ptr = g_interp_weak.lock();
    if (!ptr) {
        ptr = std::make_shared<InterpreterGuard>();
        g_interp_weak = ptr;
    }
    return ptr;
}

} // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MLXEngine::Impl {
    std::shared_ptr<InterpreterGuard> interp_guard;
    py::object model;
    py::object tokenizer;
    std::string model_dir;

    explicit Impl(const std::string& dir) : model_dir(dir) {
        interp_guard = acquire_interpreter();

        try {
            py::module_ mlx_lm = py::module_::import("mlx_lm");

            std::cout << "[MLXEngine] Loading model from " << model_dir << "..." << std::endl;

            py::tuple loaded = mlx_lm.attr("load")(model_dir);
            model     = loaded[0];
            tokenizer = loaded[1];

            std::cout << "[MLXEngine] Model loaded successfully." << std::endl;
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(std::string("[MLXEngine] Python error during load: ") + e.what());
        }
    }

    std::string generate_text(const std::string& prompt,
                              const GenerationConfig& config,
                              const std::function<bool(const std::string&)>& token_callback) {
        try {
            py::module_ mlx_lm = py::module_::import("mlx_lm");

            std::string result;

            if (token_callback) {
                // stream_generate yields one decoded string segment at a time.
                py::object generator = mlx_lm.attr("stream_generate")(
                    model,
                    tokenizer,
                    prompt,
                    py::arg("max_tokens") = config.max_tokens,
                    py::arg("temp")       = config.temperature
                );

                for (py::handle chunk : generator) {
                    // Each yielded object is a GenerationResult; .text holds the new piece.
                    std::string piece = chunk.attr("text").cast<std::string>();
                    result += piece;
                    if (!piece.empty() && !token_callback(piece)) {
                        break;
                    }
                }
            } else {
                // Non-streaming path: call generate() directly for a single string result.
                py::object output = mlx_lm.attr("generate")(
                    model,
                    tokenizer,
                    py::arg("prompt")     = prompt,
                    py::arg("max_tokens") = config.max_tokens,
                    py::arg("temp")       = config.temperature,
                    py::arg("verbose")    = false
                );
                result = output.cast<std::string>();
            }

            return result;
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(std::string("[MLXEngine] Python error during generate: ") + e.what());
        }
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MLXEngine::MLXEngine(const std::string& model_dir)
    : pimpl_(std::make_unique<Impl>(model_dir)) {}

MLXEngine::~MLXEngine() = default;

std::string MLXEngine::generate(const std::string& prompt,
                                 const GenerationConfig& config,
                                 std::function<bool(const std::string&)> token_callback) {
    try {
        return pimpl_->generate_text(prompt, config, token_callback);
    } catch (const std::exception& e) {
        std::cerr << "[MLXEngine Error] " << e.what() << std::endl;
        return "Error during generation: " + std::string(e.what());
    }
}
