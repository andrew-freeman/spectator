# sp-ng

## Running with llama-server

1. Start llama-server (example):

   ```bash
   llama-server --model /path/to/model.gguf --port 8080
   ```

2. Run Spectator with the llama backend:

   ```bash
   export SPECTATOR_BACKEND=llama
   export LLAMA_SERVER_BASE_URL=http://127.0.0.1:8080
   # Optional: export LLAMA_SERVER_MODEL=your-model-name
   python -m spectator run --text "Hello"
   ```
