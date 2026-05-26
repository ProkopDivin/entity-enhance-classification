'''
Quick GPU benchmark for Qwen3-Embedding-4B (Transformers API).
'''

import time

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

model_name = 'Qwen/Qwen3-Embedding-4B'
task = 'Represent this article for iptc topic classification'
max_length = 512


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    '''
    Pool last non-padding token per sequence (Qwen3 embedding convention).

    :param last_hidden_states: model hidden states
    :param attention_mask: batch attention mask
    :return: pooled embeddings
    '''
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def format_instruct(*, task_description: str, text: str) -> str:
    '''
    Format text with Qwen3 instruction prefix.

    :param task_description: one-sentence task description
    :param text: input text to embed
    :return: formatted input string
    '''
    return f'Instruct: {task_description}\nQuery:{text}'


def embed_texts(
    *,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    task_description: str,
    max_length: int,
    batch_size: int = 32,
) -> Tensor:
    '''
    Encode texts to L2-normalized embeddings.

    :param model: loaded Qwen3 embedding model
    :param tokenizer: matching tokenizer
    :param texts: raw input strings
    :param task_description: instruction for embedding task
    :param max_length: max token length
    :param batch_size: inference batch size
    :return: normalized embedding matrix
    '''
    formatted = [format_instruct(task_description=task_description, text=text) for text in texts]
    all_embeddings: list[Tensor] = []

    for start in range(0, len(formatted), batch_size):
        batch = formatted[start : start + batch_size]
        batch_dict = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        batch_dict = {key: value.to(model.device) for key, value in batch_dict.items()}
        outputs = model(**batch_dict)
        pooled = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        all_embeddings.append(F.normalize(pooled, p=2, dim=1))

    return torch.cat(all_embeddings, dim=0)


print('Loading model...')

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side='left',
    cache_dir='./hf_cache',
)

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    cache_dir='./hf_cache',
)

print('Model loaded on GPU')

texts = [
    'Wikipedia graphs are useful for retrieval.',
    'Machine learning and embeddings.',
] * 100

start = time.time()
embeddings = embed_texts(
    model=model,
    tokenizer=tokenizer,
    texts=texts,
    task_description=task,
    max_length=max_length,
    batch_size=32,
)
elapsed = time.time() - start

print('Embeddings shape:', embeddings.shape)
print(f'Time: {elapsed:.2f} sec')
print('GPU memory allocated:')
print(torch.cuda.memory_allocated() / 1024**2, 'MB')
