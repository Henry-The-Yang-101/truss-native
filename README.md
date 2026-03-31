# Get started

### Clone
```bash
git clone https://github.com/Henry-The-Yang-101/truss-native.git
cd truss-native
```

### Install dependencies
```bash
brew install cmake pkg-config yaml-cpp nlohmann-json
```

### Generate build files
```bash
mkdir build && cd build
cmake ..
```

### Compile the server
make

### Download the model
```bash
# Create the directory
mkdir -p models/llama3-8b-gguf

# Download the model directly via Python
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF', filename='Meta-Llama-3-8B-Instruct-Q4_K_M.gguf', local_dir='./models/llama3-8b-gguf')"
```

# Run!
```bash
cd build
./truss_server
```
In a different terminal, make this POST request to test your prompt:
```
curl -X POST http://localhost:8080/v1/predict \

     -H "Content-Type: application/json" \

     -d '{"prompt": "what is the meaning of life?"}'
```
