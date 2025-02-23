& 'llama-server' `
    --model ./models/dp1.5.gguf `
    --port 8081 `
    --threads 8 `
    --n_ctx 4096 `
    --host 0.0.0.0 `
    | Out-Null 2>&1