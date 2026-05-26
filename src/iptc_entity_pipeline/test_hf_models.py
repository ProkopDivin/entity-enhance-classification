from transformers import AutoModel
import torch
import time

model_name = "jinaai/jina-embeddings-v5-text-small"

print("Loading model...")

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    cache_dir="./hf_cache",
).cuda()

print("Model loaded on GPU")

texts = [
    "Wikipedia graphs are useful for retrieval.",
    "Machine learning and embeddings."
] * 100

start = time.time()

embeddings = model.encode(
    texts,
    task ="classification"
)

elapsed = time.time() - start

print("Embeddings shape:", embeddings.shape)
print(f"Time: {elapsed:.2f} sec")

print("GPU memory allocated:")
print(torch.cuda.memory_allocated() / 1024**2, "MB")

