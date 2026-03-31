#include "llm_engine.h"
#include "llama.h"
#include <vector>
#include <stdexcept>
#include <iostream>

struct LLMEngine::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;

    int n_past = 0; 
    const int MAX_CONTEXT = 2048; 

    Impl(const std::string& model_path) {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99; 

        std::cout << "[LLMEngine] Loading model into Metal unified memory..." << std::endl;
        model = llama_load_model_from_file(model_path.c_str(), model_params);
        if (!model) throw std::runtime_error("Failed to load model from " + model_path);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = MAX_CONTEXT; 
        
        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("Failed to create context");
    }

    ~Impl() {
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
    }

    std::string generate_text(const std::string& prompt, const GenerationConfig& config) {
        const llama_vocab* vocab = llama_model_get_vocab(model);

        std::vector<llama_token> tokens(prompt.length() + 2);
        
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), false, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), false, true);
        }
        tokens.resize(n_tokens);

        if (n_past + tokens.size() >= MAX_CONTEXT) {
            std::cout << "\n[LLMEngine] Context limit reached. Resetting conversation memory." << std::endl;
            llama_memory_clear(llama_get_memory(ctx), true);
            n_past = 0;
        }

        if (smpl) llama_sampler_free(smpl);
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy()); 

        llama_batch batch = llama_batch_init(2048, 0, 1);
        batch.n_tokens = 0;

        for (int i = 0; i < tokens.size(); i++) {
            batch.token[batch.n_tokens] = tokens[i];
            batch.pos[batch.n_tokens] = n_past + i; 
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (i == tokens.size() - 1); 
            batch.n_tokens++;
        }

        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("Failed to decode prompt");

        std::string result = "";
        int n_decode = 0; 
        
        int n_cur = n_past + tokens.size(); 

        while (n_decode < config.max_tokens && n_cur < MAX_CONTEXT) {
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, new_token_id);

            if (llama_token_is_eog(vocab, new_token_id)) {
                n_cur++; // count the end-of-generation token in our history
                break;
            }

            char buf[128];
            int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n_chars > 0) result += std::string(buf, n_chars);

            batch.n_tokens = 0;
            batch.token[batch.n_tokens] = new_token_id;
            batch.pos[batch.n_tokens] = n_cur;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;

            if (llama_decode(ctx, batch) != 0) break;

            n_cur++;
            n_decode++;
        }
        n_past = n_cur; 

        llama_batch_free(batch); 
        return result;
    }
};

// --- LLMEngine Public Methods ---

LLMEngine::LLMEngine(const std::string& model_path) 
    : pimpl_(std::make_unique<Impl>(model_path)) {}

LLMEngine::~LLMEngine() = default;

std::string LLMEngine::generate(const std::string& prompt, const GenerationConfig& config) {
    try {
        return pimpl_->generate_text(prompt, config);
    } catch (const std::exception& e) {
        std::cerr << "[LLMEngine Error] " << e.what() << std::endl;
        return "Error during generation: " + std::string(e.what());
    }
}