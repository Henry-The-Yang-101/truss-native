#include "llama_engine.h"
#include "llama.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <stdexcept>
#include <iostream>

struct LlamaEngine::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;

    int MAX_CONTEXT;
    std::vector<llama_token> cached_tokens;
    std::vector<llama_token> token_buffer;
    std::vector<llama_token> generated_ids;
    llama_batch single_token_batch{};

    Impl(const std::string& model_path, bool flash_attention, int requested_max_context) {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;

        std::cout << "[LlamaEngine] Loading model into Metal unified memory..." << std::endl;
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) throw std::runtime_error("Failed to load model from " + model_path);

        int n_ctx_train = llama_model_n_ctx_train(model);
        if (requested_max_context > 0) {
            if (requested_max_context > n_ctx_train) {
                std::cout << "[LlamaEngine] Warning: requested max_context (" << requested_max_context 
                          << ") exceeds model's training context length (" << n_ctx_train 
                          << "). Capping to " << n_ctx_train << "." << std::endl;
                MAX_CONTEXT = n_ctx_train;
            } else {
                MAX_CONTEXT = requested_max_context;
            }
        } else {
            MAX_CONTEXT = 4096;
            if (MAX_CONTEXT > n_ctx_train) MAX_CONTEXT = n_ctx_train;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.flash_attn_type =
            flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.n_ctx = MAX_CONTEXT;
        ctx_params.n_batch = MAX_CONTEXT;

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("Failed to create context");

        cached_tokens.reserve(MAX_CONTEXT);
        token_buffer.reserve(MAX_CONTEXT);
        generated_ids.reserve(MAX_CONTEXT);
        single_token_batch = llama_batch_init(1, 0, 1);
    }

    ~Impl() {
        llama_batch_free(single_token_batch);
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
    }

    void tokenize(const std::string& text, std::vector<llama_token>& tokens) {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        tokens.resize(text.length() + 2);
        int n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
        if (n < 0) {
            tokens.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
        }
        if (n < 0) {
            tokens.clear();
            return;
        }
        tokens.resize(n);
    }

    std::string generate_text(const std::string& prompt, const GenerationConfig& config,
                              const std::function<bool(const std::string&)>& token_callback) {
        const llama_vocab* vocab = llama_model_get_vocab(model);

        tokenize(prompt, token_buffer);
        const std::vector<llama_token>& tokens = token_buffer;

        if (tokens.size() > (size_t)MAX_CONTEXT) {
            throw std::runtime_error("Prompt exceeds maximum context length (" + std::to_string(tokens.size()) + " > " + std::to_string(MAX_CONTEXT) + ")");
        }

        int common_prefix = 0;
        int max_common = (int)std::min(cached_tokens.size(), tokens.size());
        while (common_prefix < max_common && cached_tokens[common_prefix] == tokens[common_prefix]) {
            common_prefix++;
        }

        if (common_prefix < (int)cached_tokens.size()) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, common_prefix, -1);
            cached_tokens.resize(common_prefix);
        }

        if (smpl) llama_sampler_free(smpl);
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        int n_new = (int)tokens.size() - common_prefix;
        if (n_new > 0) {
            llama_batch batch = llama_batch_init(n_new, 0, 1);
            batch.n_tokens = 0;

            for (int i = 0; i < n_new; i++) {
                int pos = common_prefix + i;
                batch.token[batch.n_tokens] = tokens[pos];
                batch.pos[batch.n_tokens] = pos;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = (i == n_new - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("Failed to decode prompt");
            }
            llama_batch_free(batch);
        }

        cached_tokens = tokens;

        std::string result;
        int remaining_tokens = std::max(0, std::min(config.max_tokens, MAX_CONTEXT - static_cast<int>(tokens.size())));
        result.reserve(static_cast<size_t>(remaining_tokens) * 4);
        int n_decode = 0;
        int n_cur = (int)tokens.size();
        generated_ids.clear();

        while (n_decode < config.max_tokens && n_cur < MAX_CONTEXT) {
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token_id);

            if (llama_token_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n_chars > 0) {
                std::string piece(buf, static_cast<size_t>(n_chars));
                result += piece;
                if (token_callback && !token_callback(piece)) {
                    break;
                }
            }

            single_token_batch.n_tokens = 0;
            single_token_batch.token[0] = new_token_id;
            single_token_batch.pos[0] = n_cur;
            single_token_batch.n_seq_id[0] = 1;
            single_token_batch.seq_id[0][0] = 0;
            single_token_batch.logits[0] = true;
            single_token_batch.n_tokens = 1;

            if (llama_decode(ctx, single_token_batch) != 0) break;

            generated_ids.push_back(new_token_id);
            n_cur++;
            n_decode++;
        }

        cached_tokens.insert(cached_tokens.end(), generated_ids.begin(), generated_ids.end());

        return result;
    }
};

LlamaEngine::LlamaEngine(const std::string& model_path, bool flash_attention, int max_context)
    : pimpl_(std::make_unique<Impl>(model_path, flash_attention, max_context)) {}

LlamaEngine::~LlamaEngine() = default;

std::string LlamaEngine::generate(const std::string& prompt, const GenerationConfig& config,
                                   std::function<bool(const std::string&)> token_callback) {
    try {
        return pimpl_->generate_text(prompt, config, token_callback);
    } catch (const std::exception& e) {
        std::cerr << "[LlamaEngine Error] " << e.what() << std::endl;
        return "Error during generation: " + std::string(e.what());
    }
}

int LlamaEngine::count_tokens(const std::string& text) const {
    const llama_vocab* vocab = llama_model_get_vocab(pimpl_->model);
    std::vector<llama_token> tokens(text.length() + 2);
    int n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
    }
    return n < 0 ? 0 : n;
}

int LlamaEngine::get_max_context() const {
    return pimpl_->MAX_CONTEXT;
}

void LlamaEngine::reset_cache() {
    llama_memory_clear(llama_get_memory(pimpl_->ctx), true);
    pimpl_->cached_tokens.clear();
}
