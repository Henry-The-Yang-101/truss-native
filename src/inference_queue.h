#pragma once

#include "language_models.h"
#include <string>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

struct InferenceRequest {
    std::vector<ChatMessage> messages_payload;
    GenerationConfig config;
    std::function<bool(const std::string&)> token_callback;
    std::promise<std::string> promise;
};

class InferenceQueue {
private:
    std::queue<InferenceRequest> req_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;

public:
    void push_request(InferenceRequest req) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            req_queue_.push(std::move(req));
        }
        cv_.notify_one();
    }

    InferenceRequest pop_request() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !req_queue_.empty(); });
        InferenceRequest req = std::move(req_queue_.front());
        req_queue_.pop();
        return req;
    }
};