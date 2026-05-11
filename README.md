# Get started

### Clone
```bash
git clone https://github.com/Henry-The-Yang-101/truss-native.git
cd truss-native
```

### Install dependencies
You will need CMake, Python 3 (with development headers), and a few C++ libraries.
```bash
brew install cmake pkg-config yaml-cpp nlohmann-json python3
```
For the MLX engine, you also need the `mlx-lm` and `huggingface_hub` Python packages:
```bash
pip3 install mlx-lm huggingface_hub
```

### Generate build files
```bash
mkdir build && cd build
cmake ..
```

### Compile the server
```bash
make -j4
```

### Download and Manage Models
You can easily add or remove models using the included Python script. It supports both GGUF (llama.cpp) and safetensors (Apple MLX) formats.

```bash
# Add a new model (prompts for ID, format, and HuggingFace repo)
python3 model_manager.py add

# Remove an existing model
python3 model_manager.py remove
```
This script automatically downloads the model weights and registers them in `models/models.yaml`.

# Run!
```bash
cd build
./truss_server
```

In a different terminal, first initialize the model you want to use (using the ID you registered):
```bash
curl -X POST http://localhost:8080/v1/initialize \
     -H "Content-Type: application/json" \
     -d '{"model": "llama-3"}'
```

Then, make a chat completion request:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "llama-3",
           "messages": [
             {"role": "user", "content": "what is the meaning of life?"}
           ]
         }'
```
Instead of using curl, you could also use postman, where I have included a postman collection for you to import for your convenience.