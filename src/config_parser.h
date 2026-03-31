#include <yaml-cpp/yaml.h>
#include <string>

struct TrussConfig {
    std::string model_dir;
    int port;
};

TrussConfig load_config(const std::string& filepath) {
    YAML::Node config = YAML::LoadFile(filepath);
    TrussConfig tc;
    tc.model_dir = config["system_config"]["model_dir"].as<std::string>();
    tc.port = config["system_config"]["port"].as<int>();
    return tc;
}