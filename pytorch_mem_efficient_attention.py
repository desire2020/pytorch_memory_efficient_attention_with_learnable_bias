import math
import torch
from torch.nn import functional as F
import gc
from torch.utils.checkpoint import checkpoint

from IPython import embed
def mefficient_attention(query, key, value, query_chunk_size=128, bias=None):
    batch_size, num_q, num_heads, q_features = query.shape
    dtype = query.dtype

    def _query_chunk_attention(query, key, value, bias=None, key_chunk_size=128):
        batch_size, num_kv, num_heads, k_features = key.shape
        v_features = value.shape[-1]
        dtype = query.dtype
        key_chunk_size = min(key_chunk_size, num_kv)
        query = query / (k_features ** 0.5)

        def summarize_chunk(query, key, value, bias):
            attn_weights = torch.einsum('bqhd,bkhd->bqhk', query, key).type(dtype)
            if bias is not None:
                attn_weights = attn_weights + bias.permute(0, 2, 1, 3)
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            max_score = max_score.detach()
            attn_weights = attn_weights - max_score
            exp_weights = torch.exp(attn_weights)
            exp_values = torch.einsum('bvhf,bqhv->bqhf', value, exp_weights).type(dtype)
            return (exp_values, exp_weights.sum(dim=-1), max_score.reshape((batch_size, query.shape[1], num_heads)))


        def chunk_scanner(chunk_idx):
            key_chunk = key[:, chunk_idx: chunk_idx + key_chunk_size]
            value_chunk = value[:, chunk_idx: chunk_idx + key_chunk_size]
            bias_chunk = None if bias is None else bias[..., chunk_idx: chunk_idx + key_chunk_size]
            return checkpoint(summarize_chunk, query, key_chunk, value_chunk, bias_chunk, use_reentrant=False)


        chunk_values, chunk_weights, chunk_max = zip(*map(chunk_scanner, range(0, num_kv, key_chunk_size)))

        global_max = torch.max(torch.stack(chunk_max), dim=0, keepdim=True)[0]
        max_diffs = torch.exp(torch.stack(chunk_max) - global_max)
        chunk_values = torch.stack(chunk_values) * max_diffs.unsqueeze(-1)
        chunk_weights = torch.stack(chunk_weights) * max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = chunk_weights.sum(dim=0).unsqueeze(-1)
        return all_values / all_weights
    def chunk_scanner(chunk_idx):
        query_chunk = query[:, chunk_idx: min(chunk_idx+query_chunk_size, num_q)]
        bias_chunk = None if bias is None else bias[..., chunk_idx: min(chunk_idx+query_chunk_size, num_q), :]
        return _query_chunk_attention(query_chunk, key, value, bias=bias_chunk)

    collected = list(map(chunk_scanner, range(0, num_q, query_chunk_size)))
    res = torch.cat(collected, dim=1)
    return res

# mefficient_attention = torch.compile(mefficient_attention) Does not work yet
def basic_attention(query, key, value, dtype=torch.float32, bias=None):
    bsz, num_q, num_heads, q_features = query.shape
    bsz, num_kv, num_heads, k_features = key.shape
    attn_weights = torch.einsum('bqhd,bkhd->bqhk', query, key).type(dtype) / (k_features ** 0.5)
    if bias is not None:
        attn_weights = attn_weights + bias.permute(0, 2, 1, 3)
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum('bqhk,bkhd->bqhd', attn_weights, value).type(dtype)
    return attn_output

basic_attention_opt = torch.compile(basic_attention)

def memory_usage(device):
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)
    return peak_mem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

batch_size = 4
num_q = 1024
num_kv = 1024
num_heads = 20
features = 64

query = torch.randn(batch_size, num_q, num_heads, features, device=device).type(torch.float32).requires_grad_(True)
key = torch.randn(batch_size, num_kv, num_heads, features, device=device).type(torch.float32).requires_grad_(True)
value = torch.randn(batch_size, num_kv, num_heads, features, device=device).type(torch.float32).requires_grad_(True)
bias = torch.randn(batch_size, num_heads, num_q, num_kv, device=device).type(torch.float32).requires_grad_(True)
bias = None
# Free any memory that might be lingering around
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats(device)
import time
if bias == None:
    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_math=False, enable_mem_efficient=True
                ):
        start_mem = memory_usage(device)
        tick = time.time()
        mefficient_res = torch.nn.functional.scaled_dot_product_attention(query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        mefficient_res.sum().backward()
        tock = time.time()
        print("Test finished in", tock - tick, "Seconds. Memory used by official efficient attention: ", memory_usage(device) - start_mem, "MB")
        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()
        if bias is not None:
            bias.grad.zero_()

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats(device)

# Test for mefficient_attention
start_mem = memory_usage(device)
tick = time.time()
mefficient_res = mefficient_attention(query, key, value, bias=bias)
mefficient_res.sum().backward()
tock = time.time()
print("Test finished in", tock - tick, "Seconds. Memory used by efficient attention: ", memory_usage(device) - start_mem, "MB")

mefficient_query_grad = query.grad.clone()
mefficient_key_grad = key.grad.clone()
mefficient_value_grad = value.grad.clone()
if bias is not None:
    mefficient_bias_grad = bias.grad.clone()

basic_res = basic_attention_opt(query, key, value, bias=bias)
basic_res.sum().backward()
query.grad.zero_()
key.grad.zero_()
value.grad.zero_()
if bias is not None:
    bias.grad.zero_()

# Free any memory that might be lingering around
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats(device)

# Test for basic_attention
start_mem = memory_usage(device)
tick = time.time()
basic_res = basic_attention_opt(query, key, value, bias=bias)
basic_res.sum().backward()
tock = time.time()
print("Test finished in", tock - tick, "Seconds. Memory used by basic attention: ", memory_usage(device) - start_mem, "MB")

print("Max absolute difference in gradients (query): ", torch.max(torch.abs(mefficient_query_grad - query.grad)).item())
print("Mean absolute difference in gradients (query): ", torch.mean(torch.abs(mefficient_query_grad - query.grad)).item())

print("Max absolute difference in gradients (key): ", torch.max(torch.abs(mefficient_key_grad - key.grad)).item())
print("Mean absolute difference in gradients (key): ", torch.mean(torch.abs(mefficient_key_grad - key.grad)).item())

print("Max absolute difference in gradients (value): ", torch.max(torch.abs(mefficient_value_grad - value.grad)).item())
print("Mean absolute difference in gradients (value): ", torch.mean(torch.abs(mefficient_value_grad - value.grad)).item())

if bias is not None:

    print("Max absolute difference in gradients (bias): ", torch.max(torch.abs(mefficient_bias_grad - bias.grad)).item())
    print("Mean absolute difference in gradients (bias): ", torch.mean(torch.abs(mefficient_bias_grad - bias.grad)).item())

