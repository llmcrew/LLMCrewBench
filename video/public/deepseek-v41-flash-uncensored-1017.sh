vllm serve \
  --served-model-name=deepseek-v4.1-flash-uncensored-1017 \
  --model=/data \
  --tensor-parallel-size=4 \
  --attention-config='{
    "backend":"FLASHINFER_MLA_SPARSE_DSV41",
    "indexer_kv_dtype":"mxfp4",
    "indexer_sparse_logits":true
  }' \
  --engram-config='{"cpu_offload":true}' \
  --mm-encoder-tp-mode=data \
  --kv-cache-dtype=fp8 \
  --max-model-len=1048576 \
  --max-num-seqs=128 \
  --max-num-batched-tokens=65536 \
  --max-cudagraph-capture-size=1024 \
  --gpu-memory-utilization=0.90 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode=deepseek_v41 \
  --reasoning-parser=deepseek_v41 \
  --tool-call-parser=deepseek_v41 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs='{
    "thinking":true,
    "reasoning_effort":"max"
  }' \
  --speculative-config='{
    "method":"dspark",
    "num_speculative_tokens":5,
    "draft_sample_method":"probabilistic",
    "rejection_sample_method":"block",
    "enable_adaptive_verification":true
  }' \
  --host=0.0.0.0 \
  --port=8000
