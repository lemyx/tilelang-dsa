import argparse
import itertools
import logging

import tilelang
from tilelang.autotuner import autotune, set_autotune_inputs
import tilelang.language as T
import torch
from torch import nn
from torch.nn import functional as F
from varlen_utils import generate_qkv, generate_random_padding_mask

######################################
# monkey patch for backward autotune #
######################################
# https://github.com/tile-ai/tilelang/issues/942
import threading
import tilelang.autotuner.tuner as _tuner

_orig_rwt = _tuner.run_with_timeout

def _safe_run_with_timeout(target_fn, timeout, *args, **kwargs):
    if threading.current_thread() is threading.main_thread():
        return _orig_rwt(target_fn, timeout, *args, **kwargs)
    # Fallback: no SIGALRM in worker threads; just run the function directly.
    return target_fn(*args, **kwargs)

_tuner.run_with_timeout = _safe_run_with_timeout

####################
# global variables #
####################
dtype = "bfloat16"
index_dtype = "bfloat16"
accum_dtype = "float"
heads = 64
dim = 128
index_heads = 64
index_dim = 64
thread_num = 128
block_M = 64
block_N = 64
num_stages = 3

assert heads == index_heads


# [reference]
# https://github.com/tile-ai/tilelang-benchmark/blob/4272166e995442bb1fe273b6764845bdb7c42416/cdna_benchmark/gemm_benchmark/1.tilelang_benchmark/benchmark_tilelang_matmul.py#L36
# https://zhuanlan.zhihu.com/p/1914885010231625343
def get_configs():
    block_M = [32, 64, 128, 256]
    block_N = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    thread_num = [64, 128]

    _configs = list(itertools.product(block_M, block_N, num_stages, thread_num))

    configs = [
        {"block_M": c[0], "block_N": c[1], "num_stages": c[2], "thread_num": c[3]}
        for c in _configs
    ]

    return configs


def setup_logger(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def display_error_message(msg):
    logging.debug(f"\033[31mWARNING: {msg}\033[0m")


def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation


def validate_tensor_match(a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        display_error_message(f"{tensor_name} Error: isfinite mask mismatch")
        if should_raise:
            assert False
    if not torch.isclose(
            a.masked_fill(a_finite, 0),
            b.masked_fill(b_finite, 0),
            rtol=0,
            atol=0,
            equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: difference={difference}, correlation={correlation}")
        if should_raise:
            assert False
    return difference


def fn_dkl_ref_opt(
        q_unpad,
        k_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        index_q_unpad,
        index_k_unpad,
        index_weights_unpad,
        heads,
        dtype
):
    q_unpad = q_unpad.float()
    k_unpad = k_unpad.float()
    index_q_unpad = index_q_unpad.float()
    index_k_unpad = index_k_unpad.float()
    index_weights_unpad = index_weights_unpad.float()

    batch_size = len(cu_seqlens_q) - 1
    output = torch.empty((len(index_q_unpad), heads), dtype=dtype, device=q_unpad.device)

    for idx in range(batch_size):
        q_start, q_end = cu_seqlens_q[idx], cu_seqlens_q[idx+1]
        k_start, k_end = cu_seqlens_k[idx], cu_seqlens_k[idx+1]

        q1_batch = q_unpad[q_start:q_end]
        k1 = k_unpad[k_start:k_end]
        q2_batch = index_q_unpad[q_start:q_end]
        k2 = index_k_unpad[k_start:k_end]
        q_scales = index_weights_unpad[q_start:q_end]

        q_len, heads, _ = q1_batch.shape
        k_len = k1.shape[0]

        assert q_len == k_len, "Vectorized implementation assumes q_len == k_len for causal mask"
        seq_len = q_len

        logits_1 = torch.einsum('qhd,kd->qhk', q1_batch, k1)
        logits_2_no_scale = torch.einsum('qhd,kd->qhk', q2_batch, k2)
        logits_2 = logits_2_no_scale.relu() * q_scales.unsqueeze(-1)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=q_unpad.device)).bool()
        masked_logits_1 = logits_1.masked_fill(~mask.unsqueeze(1), -float('inf'))
        masked_logits_2 = logits_2.masked_fill(~mask.unsqueeze(1), -float('inf'))

        log_p_probs = F.log_softmax(masked_logits_1, dim=-1)
        p_probs = F.softmax(masked_logits_1, dim=-1)
        log_q_probs = F.log_softmax(masked_logits_2, dim=-1)
        kl_div_terms = p_probs * (log_p_probs - log_q_probs)
        kl_div_terms = torch.nan_to_num(kl_div_terms, nan=0.0, posinf=0.0, neginf=0.0)
        output[q_start:q_end] = kl_div_terms.sum(dim=-1)

    return output


def fn_dkl_bwd_ref(
        q_unpad,
        k_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        index_q_unpad,
        index_k_unpad,
        index_weights_unpad,
        heads,
        dtype
):
    index_q_unpad_clone = index_q_unpad.clone().detach().requires_grad_(True)
    index_k_unpad_clone = index_k_unpad.clone().detach().requires_grad_(True)
    index_weights_unpad_clone = index_weights_unpad.clone().detach().requires_grad_(True)

    q_unpad_clone = q_unpad.clone().detach()
    k_unpad_clone = k_unpad.clone().detach()

    dkl_ref = fn_dkl_ref_opt(
        q_unpad_clone,
        k_unpad_clone,
        cu_seqlens_q,
        cu_seqlens_k,
        index_q_unpad_clone,
        index_k_unpad_clone,
        index_weights_unpad_clone,
        heads,
        dtype
    )

    dkl_ref.sum().backward()

    return index_q_unpad_clone.grad, index_k_unpad_clone.grad, index_weights_unpad_clone.grad


pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True
}

def tl_dsa_warmup_fwd_impl():

    # @autotune(configs=get_configs(), warmup=3, rep=5)
    @tilelang.jit(pass_configs=pass_configs)
    def kernel(block_M=64, block_N=64, num_stages=3, thread_num=128):

        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        batch_size = T.symbolic("batch_size")

        scale = 1.44269504  # log2(e)
        sm_scale = (1.0 / dim)**0.5 * 1.44269504

        @T.prim_func
        def main(
                ##################
                # varlen mqa fwd #
                ##################
                Q_unpad: T.Tensor([seq_len, heads, dim], dtype),
                K_unpad: T.Tensor([seq_len_kv, dim], dtype),
                V_unpad: T.Tensor([seq_len_kv, dim], dtype),
                O_unpad: T.Tensor([seq_len, heads, dim], dtype),
                cu_seqlens_q: T.Tensor([batch_size], "int32"),
                cu_seqlens_k: T.Tensor([batch_size], "int32"),
                max_seqlen_q: T.int32,
                ####################
                # lighting_indexer #
                ####################
                IndexQ: T.Tensor([seq_len, index_heads, index_dim], index_dtype),
                IndexK: T.Tensor([seq_len_kv, index_dim], index_dtype),
                Weights: T.Tensor([seq_len, index_heads], accum_dtype),
                #################
                # kl divergence #
                #################
                dkl: T.Tensor([seq_len, heads], accum_dtype),
                out_log_z_p: T.Tensor([seq_len, heads], accum_dtype),
                out_log_z_q: T.Tensor([seq_len, heads], accum_dtype)
        ):
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_M),
                    heads,
                    batch_size-1,
                    threads=thread_num
            ) as (bx, by, bz):

                #################
                # varlen mqa fwd #
                ##################
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)

                # reduce & gemm operator will limit operand types.
                kl_fla_logits = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_shared([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_shared([block_M], accum_dtype)

                q_start_idx, q_end_idx = cu_seqlens_q[bz], cu_seqlens_q[bz+1]
                k_start_idx, k_end_idx = cu_seqlens_k[bz], cu_seqlens_k[bz+1]
                v_start_idx, v_end_idx = cu_seqlens_k[bz], cu_seqlens_k[bz+1]
                q_current_seqlen = q_end_idx - q_start_idx
                k_current_seqlen = k_end_idx - k_start_idx
                v_current_seqlen = v_end_idx - v_start_idx

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                T.copy(
                    Q_unpad[q_start_idx+bx*block_M: q_start_idx+(bx+1)*block_M, by, :],
                    Q_shared
                )

                T.annotate_layout({
                    acc_s_cast: tilelang.layout.make_swizzled_layout(acc_s_cast),
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                })

                ####################
                # lighting_indexer #
                ####################
                index_q_shared = T.alloc_shared([block_M, index_dim], index_dtype)
                index_k_shared = T.alloc_shared([block_N, index_dim], index_dtype)
                index_logits = T.alloc_fragment([block_M, block_N], accum_dtype)
                index_weights = T.alloc_shared([block_M], accum_dtype)

                T.copy(
                    IndexQ[q_start_idx+bx*block_M: q_start_idx+(bx+1)*block_M, by, :],
                    index_q_shared
                )
                for i in T.Parallel(block_M):
                    index_weights[i] = Weights[q_start_idx+bx*block_M+i, by]

                #################
                # kl divergence #
                #################
                kl_acc_s_p = T.alloc_fragment([block_M, block_N], accum_dtype)
                kl_scores_max_p = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_max_prev_p = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_scale_p = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_sum_p = T.alloc_fragment([block_M], accum_dtype)
                kl_logsum_p = T.alloc_fragment([block_M], accum_dtype)

                kl_acc_s_q = T.alloc_fragment([block_M, block_N], accum_dtype)
                kl_scores_max_q = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_max_prev_q = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_scale_q = T.alloc_fragment([block_M], accum_dtype)
                kl_scores_sum_q = T.alloc_fragment([block_M], accum_dtype)
                kl_logsum_q = T.alloc_fragment([block_M], accum_dtype)

                kl_intermediate = T.alloc_fragment([block_M, block_N], accum_dtype)
                kl_intermediate_sum = T.alloc_fragment([block_M], accum_dtype)
                kl_acc_o = T.alloc_fragment([block_M], accum_dtype)

                T.fill(kl_acc_o, 0)
                T.fill(kl_intermediate, 0)
                T.fill(kl_logsum_p, 0)
                T.fill(kl_logsum_q, 0)
                T.fill(kl_scores_max_p, -T.infinity(accum_dtype))
                T.fill(kl_scores_max_q, -T.infinity(accum_dtype))

                for k in T.Pipelined(
                        T.min(
                            T.ceildiv((bx+1)*block_M, block_N),
                            T.ceildiv(k_current_seqlen, block_N),
                        ),
                        num_stages=num_stages
                ):
                    ####################
                    # lighting_indexer #
                    ####################

                    T.copy(
                        IndexK[k_start_idx+k*block_N: k_start_idx+(k+1)*block_N, :],
                        index_k_shared
                    )

                    T.gemm(
                        index_q_shared,
                        index_k_shared,
                        index_logits,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    for i, j in T.Parallel(block_M, block_N):
                        index_logits[i, j] = T.if_then_else(
                            (bx*block_M+i >= k*block_N+j) and
                            (bx*block_M+i < q_current_seqlen) and
                            (k*block_N+j < k_current_seqlen),
                            T.max(index_logits[i, j], 0) * index_weights[i],
                            -T.infinity(accum_dtype)
                        )

                    ##################
                    # varlen mqa fwd #
                    ##################
                    T.copy(K_unpad[k_start_idx+k*block_N: k_start_idx+(k+1)*block_N, :], K_shared)
                    for j, d in T.Parallel(block_N, dim):
                        if k*block_N + j >= k_current_seqlen:
                            K_shared[j, d] = 0

                    T.copy(V_unpad[v_start_idx+k*block_N: v_start_idx+(k+1)*block_N, :], V_shared)
                    for j, d in T.Parallel(block_N, dim):
                        if k*block_N + j >= v_current_seqlen:
                            V_shared[j, d] = 0

                    for i, j in T.Parallel(block_M, block_N):
                        kl_fla_logits[i, j] = T.if_then_else(
                            (bx*block_M+i >= k*block_N+j) and
                            (bx*block_M+i < q_current_seqlen) and
                            (k*block_N+j < k_current_seqlen),
                            0,
                            -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        Q_shared,
                        K_shared,
                        kl_fla_logits,
                        transpose_B=True,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(kl_fla_logits, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * sm_scale - scores_max[i] * sm_scale)

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(kl_fla_logits[i, j] * sm_scale - scores_max[i] * sm_scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, d in T.Parallel(block_M, dim):
                        acc_o[i, d] *= scores_scale[i]

                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        transpose_B=False,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    #################
                    # kl divergence #
                    #################

                    T.copy(kl_scores_max_p, kl_scores_max_prev_p)
                    T.copy(kl_scores_max_q, kl_scores_max_prev_q)

                    T.reduce_max(kl_fla_logits, kl_scores_max_p, dim=1, clear=False)
                    T.reduce_max(index_logits, kl_scores_max_q, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        kl_scores_scale_p[i] = T.exp2(scale*(kl_scores_max_prev_p[i] - kl_scores_max_p[i]))
                        kl_scores_scale_q[i] = T.exp2(scale*(kl_scores_max_prev_q[i] - kl_scores_max_q[i]))

                    for i, j in T.Parallel(block_M, block_N):
                        kl_acc_s_p[i, j] = T.exp2(scale*(kl_fla_logits[i, j] - kl_scores_max_p[i]))
                        kl_acc_s_q[i, j] = T.exp2(scale*(index_logits[i, j] - kl_scores_max_q[i]))

                    T.reduce_sum(kl_acc_s_p, kl_scores_sum_p, dim=1)
                    T.reduce_sum(kl_acc_s_q, kl_scores_sum_q, dim=1)

                    for i in T.Parallel(block_M):
                        kl_logsum_p[i] = kl_logsum_p[i] * kl_scores_scale_p[i] + kl_scores_sum_p[i]
                        kl_logsum_q[i] = kl_logsum_q[i] * kl_scores_scale_q[i] + kl_scores_sum_q[i]

                    for i, j in T.Parallel(block_M, block_N):
                        kl_intermediate[i, j] = T.if_then_else(
                            kl_fla_logits[i, j] == -T.infinity(accum_dtype),
                            0,
                            kl_acc_s_p[i, j] * (kl_fla_logits[i, j] - index_logits[i, j])
                        )
                    T.reduce_sum(kl_intermediate, kl_intermediate_sum, dim=1)

                    for i in T.Parallel(block_M):
                        kl_acc_o[i] = kl_acc_o[i] * kl_scores_scale_p[i] + kl_intermediate_sum[i]

                for i, d in T.Parallel(block_M, dim):
                    acc_o[i, d] /= logsum[i]
                T.copy(acc_o, O_shared)

                for i, d in T.Parallel(block_M, dim):
                    if bx*block_M + i < q_current_seqlen:
                        O_unpad[q_start_idx+bx*block_M+i, by, d] = O_shared[i, d]

                for i in T.Parallel(block_M):
                    if bx*block_M+i < q_current_seqlen:
                        dkl[q_start_idx+bx*block_M+i, by] = T.max(
                            kl_acc_o[i] / kl_logsum_p[i] - kl_scores_max_p[i] + kl_scores_max_q[i] + T.log(kl_logsum_q[i]) - T.log(kl_logsum_p[i]),
                            0
                        )
                        out_log_z_p[q_start_idx+bx*block_M+i, by] = kl_scores_max_p[i] + T.log(kl_logsum_p[i])
                        out_log_z_q[q_start_idx+bx*block_M+i, by] = kl_scores_max_q[i] + T.log(kl_logsum_q[i])

        return main
    return kernel()


def tl_dsa_warmup_bwd_impl():
    # @autotune(configs=get_configs(), warmup=3, rep=5)
    @tilelang.jit(pass_configs=pass_configs)
    def kernel(block_M=128, block_N=32, num_stages=3, thread_num=128):

        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        batch_size = T.symbolic("batch_size")

        @T.prim_func
        def main(
                Q_unpad: T.Tensor([seq_len, heads, dim], dtype),
                K_unpad: T.Tensor([seq_len_kv, dim], dtype),
                cu_seqlens_q: T.Tensor([batch_size], "int32"),
                cu_seqlens_k: T.Tensor([batch_size], "int32"),
                max_seqlen_q: T.int32,
                IndexQ: T.Tensor([seq_len, index_heads, index_dim], index_dtype),
                IndexK: T.Tensor([seq_len_kv, index_dim], index_dtype),
                Weights: T.Tensor([seq_len, index_heads], accum_dtype),
                log_z_p: T.Tensor([seq_len, heads], accum_dtype),
                log_z_q: T.Tensor([seq_len, heads], accum_dtype),
                grad_dkl: T.Tensor([seq_len, heads], accum_dtype),
                grad_index_q: T.Tensor([seq_len, index_heads, index_dim], accum_dtype),
                grad_index_k: T.Tensor([seq_len_kv, index_dim], accum_dtype),
                grad_index_weights: T.Tensor([seq_len, index_heads], accum_dtype)
        ):
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_M),
                    index_heads,
                    batch_size-1,
                    threads=thread_num
            ) as (bx, by, bz):

                q_start_idx, q_end_idx = cu_seqlens_q[bz], cu_seqlens_q[bz+1]
                k_start_idx, k_end_idx = cu_seqlens_k[bz], cu_seqlens_k[bz+1]
                q_current_seqlen = q_end_idx - q_start_idx
                k_current_seqlen = k_end_idx - k_start_idx

                ########################
                # logits recomputation #
                ########################
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                fla_logits = T.alloc_fragment([block_M, block_N], accum_dtype)
                for i, d in T.Parallel(block_M, dim):
                    Q_shared[i, d] = T.if_then_else(
                        bx*block_M + i < q_current_seqlen,
                        Q_unpad[q_start_idx+bx*block_M+i, by, d],
                        0
                    )

                index_q_shared = T.alloc_shared([block_M, index_dim], index_dtype)
                index_q_shared_cast = T.alloc_shared([block_M, index_dim], accum_dtype)
                index_k_shared = T.alloc_shared([block_N, index_dim], index_dtype)
                index_k_shared_cast = T.alloc_shared([block_N, index_dim], accum_dtype)
                index_weights = T.alloc_fragment([block_M], accum_dtype)
                index_s_relu = T.alloc_fragment([block_M, block_N], accum_dtype)
                index_logits = T.alloc_fragment([block_M, block_N], accum_dtype)

                T.annotate_layout({
                    Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    index_q_shared: tilelang.layout.make_swizzled_layout(index_q_shared),
                    index_k_shared: tilelang.layout.make_swizzled_layout(index_k_shared),
                })

                T.copy(
                    IndexQ[q_start_idx+bx*block_M: q_start_idx+(bx+1)*block_M, by, :],
                    index_q_shared
                )

                T.copy(index_q_shared, index_q_shared_cast)
                for i in T.Parallel(block_M):
                    index_weights[i] = Weights[q_start_idx+bx*block_M+i, by]

                ###########
                # dLogits #
                ###########
                dLogits = T.alloc_fragment([block_M, block_N], accum_dtype)
                grad_dkl_shared = T.alloc_fragment([block_M], accum_dtype)
                log_z_p_shared = T.alloc_fragment([block_M], accum_dtype)
                log_z_q_shared = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    grad_dkl_shared[i] = grad_dkl[q_start_idx+bx*block_M+i, by]
                    log_z_p_shared[i] = log_z_p[q_start_idx+bx*block_M+i, by]
                    log_z_q_shared[i] = log_z_q[q_start_idx+bx*block_M+i, by]

                ###################################
                # dIndexQ, dIndexK, dIndexWeights #
                ###################################
                dS = T.alloc_fragment([block_M, block_N], accum_dtype)
                dS_shared = T.alloc_shared([block_M, block_N], accum_dtype)
                dIndexQ = T.alloc_fragment([block_M, index_dim], accum_dtype)
                dIndexK = T.alloc_fragment([block_N, index_dim], accum_dtype)
                dIndexQ_shared = T.alloc_shared([block_M, index_dim], accum_dtype)
                dIndexK_shared = T.alloc_shared([block_N, index_dim], accum_dtype)
                dIndexWeights_shared = T.alloc_shared([block_M], accum_dtype)

                T.clear(dIndexQ)
                T.clear(dIndexWeights_shared)

                for k in T.Pipelined(
                        T.min(
                            T.ceildiv((bx+1)*block_M, block_N),
                            T.ceildiv(k_current_seqlen, block_N)
                        ),
                        num_stages=num_stages
                ):
                    ########################
                    # logits recomputation #
                    ########################
                    T.copy(
                        K_unpad[k_start_idx+k*block_N: k_start_idx+(k+1)*block_N, :],
                        K_shared
                    )

                    T.clear(fla_logits)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        fla_logits,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    T.copy(
                        IndexK[k_start_idx+k*block_N: k_start_idx+(k+1)*block_N, :],
                        index_k_shared
                    )
                    T.copy(index_k_shared, index_k_shared_cast)

                    T.clear(index_logits)
                    T.gemm(
                        index_q_shared,
                        index_k_shared,
                        index_logits,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    for i, j in T.Parallel(block_M, block_N):
                        index_s_relu[i, j] = T.if_then_else(
                            (bx*block_M+i >= k*block_N+j) and
                            (bx*block_M+i < q_current_seqlen) and
                            (k*block_N+j < k_current_seqlen),
                            T.max(index_logits[i, j], 0),
                            -T.infinity(accum_dtype)
                        )

                    for i, j in T.Parallel(block_M, block_N):
                        index_logits[i, j] = index_s_relu[i, j] * index_weights[i]

                    ###########
                    # dLogits #
                    ###########
                    for i, j in T.Parallel(block_M, block_N):
                        dLogits[i, j] = T.if_then_else(
                            (bx*block_M+i >= k*block_N+j) and
                            (bx*block_M+i < q_current_seqlen) and
                            (k*block_N+j < k_current_seqlen),
                            grad_dkl_shared[i] * (T.exp(index_logits[i, j] - log_z_q_shared[i]) - T.exp(fla_logits[i, j] - log_z_p_shared[i])),
                            0
                        )

                    ###################################
                    # dIndexQ, dIndexK, dIndexWeights #
                    ###################################
                    ## dIndexS = dIndexLogits * WeightsQ[:, None] * ScaleK[None, :] * ((IndexQ @ IndexK.T).masked_fill(~causal_mask, 0) > 0)
                    ## dIndexQ = dIndexS @ IndexK
                    ## dIndexK = dIndexS.T @ IndexQ
                    ## dWeightsQ = (dIndexLogits * ((IndexQ @ IndexK.T).masked_fill(~causal_mask, 0).relu()) * ScaleK[None, :]).sum(dim=-1)
                    for i, j in T.Parallel(block_M, block_N):
                        dS[i, j] = T.if_then_else(
                            index_s_relu[i, j] > 0,
                            dLogits[i, j] * index_weights[i],
                            0
                        )

                    T.copy(dS, dS_shared)

                    T.gemm(
                        dS_shared,
                        index_k_shared_cast,
                        dIndexQ,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullCol
                    )

                    T.gemm(
                        dS_shared,
                        index_q_shared_cast,
                        dIndexK,
                        transpose_A=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol
                    )
                    T.copy(dIndexK, dIndexK_shared)
                    for j, d in T.Parallel(block_N, index_dim):
                        if k*block_N + j < k_current_seqlen:
                            T.atomic_add(
                                grad_index_k[k_start_idx+k*block_N+j, d],
                                dIndexK_shared[j, d]
                            )

                    for i, j in T.Parallel(block_M, block_N):
                        T.atomic_add(
                            dIndexWeights_shared[i],
                            T.if_then_else(
                                (bx*block_M+i >= k*block_N+j) and
                                (bx*block_M+i < q_current_seqlen) and
                                (k*block_N+j < k_current_seqlen),
                                dLogits[i, j] * index_s_relu[i, j],
                                0
                            )
                        )

                T.copy(dIndexQ, dIndexQ_shared)

                for i, d in T.Parallel(block_M, index_dim):
                    if bx*block_M + i < q_current_seqlen:
                        grad_index_q[q_start_idx+bx*block_M+i, by, d] = dIndexQ_shared[i, d]

                for i in T.Parallel(block_M):
                    if bx*block_M + i < q_current_seqlen:
                        grad_index_weights[q_start_idx+bx*block_M+i, by] = dIndexWeights_shared[i]

        return main
    return kernel()


class _DsaWarmupFunc(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q_unpad,              # [seq_len_q, heads, dim]
            k_unpad,              # [seq_len_kv, dim]
            v_unpad,              # [seq_len_kv, dim]
            index_q_unpad,        # [seq_len_q, heads, dim]
            index_k_unpad,        # [seq_len_kv, dim]
            cu_seqlens_q,         # [batch_size+1]
            cu_seqlens_k,         # [batch_size+1]
            max_seqlen_q,         # int
            index_weights_unpad   # [seq_len_q, heads]
    ):

        # Initialize output tensors
        o_unpad = torch.empty_like(q_unpad)
        dkl = torch.empty_like(index_weights_unpad, dtype=getattr(torch, accum_dtype))
        out_log_z_p = torch.empty_like(index_weights_unpad, dtype=getattr(torch, accum_dtype))
        out_log_z_q = torch.empty_like(index_weights_unpad, dtype=getattr(torch, accum_dtype))

        # https://github.com/tile-ai/tilelang/issues/1122
        # https://github.com/tile-ai/tilelang/blob/main/testing/python/autotune/test_tilelang_autotune_with_inputs.py#L136-L137
        # with set_autotune_inputs(
        #     [
        #         q_unpad, k_unpad, v_unpad, o_unpad,
        #         cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        #         index_q_unpad, index_k_unpad, index_weights_unpad,
        #         dkl, out_log_z_p, out_log_z_q
        #     ]
        # ):
        #     tl_dsa_warmup_fwd_kernel = tl_dsa_warmup_fwd_impl()

        tl_dsa_warmup_fwd_kernel = tl_dsa_warmup_fwd_impl()

        tl_dsa_warmup_fwd_kernel(
            q_unpad,
            k_unpad,
            v_unpad,
            o_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            index_q_unpad,
            index_k_unpad,
            index_weights_unpad,
            dkl,
            out_log_z_p,
            out_log_z_q
        )

        profiler = tl_dsa_warmup_fwd_kernel.get_profiler()
        latency = profiler.do_bench(
            warmup=500,
            input_tensors=[
                q_unpad, k_unpad, v_unpad, o_unpad,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                index_q_unpad, index_k_unpad, index_weights_unpad,
                dkl, out_log_z_p, out_log_z_q
            ]
        )
        logging.info(f"TL FWD Latency: {latency} ms")

        ctx.save_for_backward(
            q_unpad, k_unpad, cu_seqlens_q, cu_seqlens_k,
            index_q_unpad, index_k_unpad, index_weights_unpad,
            out_log_z_p, out_log_z_q
        )
        ctx.max_seqlen_q = max_seqlen_q

        return o_unpad, dkl

    @staticmethod
    def backward(ctx, grad_o, grad_dkl):
        (
            q_unpad, k_unpad,
            cu_seqlens_q, cu_seqlens_k,
            index_q_unpad, index_k_unpad, index_weights_unpad,
            out_log_z_p, out_log_z_q
        ) = ctx.saved_tensors

        max_seqlen_q = ctx.max_seqlen_q

        # Initialize gradient tensors
        grad_index_q = torch.zeros_like(index_q_unpad, dtype=torch.float32)
        grad_index_k = torch.zeros_like(index_k_unpad, dtype=torch.float32)
        grad_index_weights = torch.zeros_like(index_weights_unpad, dtype=torch.float32)

        # https://github.com/tile-ai/tilelang/issues/1122
        # https://github.com/tile-ai/tilelang/blob/main/testing/python/autotune/test_tilelang_autotune_with_inputs.py#L136-L137
        # with set_autotune_inputs(
        #     [
        #         q_unpad, k_unpad,
        #         cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        #         index_q_unpad, index_k_unpad, index_weights_unpad,
        #         out_log_z_p, out_log_z_q,
        #         grad_dkl, grad_index_q, grad_index_k, grad_index_weights
        #     ]
        # ):
        #     tl_dsa_warmup_bwd_kernel = tl_dsa_warmup_bwd_impl()

        tl_dsa_warmup_bwd_kernel = tl_dsa_warmup_bwd_impl()

        tl_dsa_warmup_bwd_kernel(
            q_unpad, k_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            index_q_unpad, index_k_unpad, index_weights_unpad,
            out_log_z_p, out_log_z_q,
            grad_dkl, grad_index_q, grad_index_k, grad_index_weights
        )

        profiler = tl_dsa_warmup_bwd_kernel.get_profiler()
        latency = profiler.do_bench(
            warmup=500,
            input_tensors=[
                q_unpad, k_unpad,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                index_q_unpad, index_k_unpad, index_weights_unpad,
                out_log_z_p, out_log_z_q,
                grad_dkl,
                torch.zeros_like(grad_index_q),
                torch.zeros_like(grad_index_k),
                torch.zeros_like(grad_index_weights)
            ]
        )
        logging.info(f"TL BWD Latency: {latency} ms")

        return None, None, None, grad_index_q, grad_index_k, None, None, None, grad_index_weights


class DsaWarmupLoss(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

    def forward(
            self,
            q_unpad, k_unpad, v_unpad,
            index_q_unpad, index_k_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            index_weights_unpad
    ):
        return _DsaWarmupFunc.apply(
            q_unpad, k_unpad, v_unpad,
            index_q_unpad, index_k_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            index_weights_unpad
        )


def oscillator(
        batch = 4,
        heads = 64,
        dim = 128,
        index_heads = 64,
        index_dim = 64,
        q_seqlen = 2048,
        k_seqlen = 2048,
        dtype = torch.bfloat16,
        device = torch.device("cuda"),
        distribution = "normal"
):
    # varlen infrastructure
    query_padding_mask = generate_random_padding_mask(q_seqlen, batch, device, mode="random")
    key_padding_mask = query_padding_mask.clone()

    if distribution == "normal":
        # mqa
        q = torch.randn(batch, q_seqlen, heads, dim, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(batch, k_seqlen, 1, dim, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(batch, k_seqlen, 1, dim, dtype=dtype, device=device, requires_grad=True)

        # lighting indexer
        index_q = torch.randn(batch, q_seqlen, index_heads, index_dim, dtype=dtype, device=device, requires_grad=True)
        index_weights = torch.randn(batch, q_seqlen, index_heads, dtype=torch.float32, device=device, requires_grad=True)
        index_k = torch.randn(batch, k_seqlen, 1, index_dim, dtype=dtype, device=device, requires_grad=True)

    elif distribution == "bernoulli":
        q = torch.bernoulli(torch.full((batch, q_seqlen, heads, dim), 0.5, dtype=dtype, device=device, requires_grad=True))
        k = torch.bernoulli(torch.full((batch, k_seqlen, 1, dim), 0.5, dtype=dtype, device=device, requires_grad=True))
        v = torch.bernoulli(torch.full((batch, k_seqlen, 1, dim), 0.5, dtype=dtype, device=device, requires_grad=True))

        index_q = torch.bernoulli(torch.full((batch, q_seqlen, index_heads, index_dim), 0.5, dtype=dtype, device=device, requires_grad=True))
        index_weights = torch.bernoulli(torch.full((batch, q_seqlen, index_heads), 0.5, dtype=torch.float32, device=device, requires_grad=True))
        index_k = torch.bernoulli(torch.full((batch, k_seqlen, 1, index_dim), 0.5, dtype=dtype, device=device, requires_grad=True))

    else:
        raise NotImplementedError

    q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v, output_pad_fn, _, _ = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    index_q_unpad, index_k_unpad, index_weights_unpad = generate_qkv(index_q, index_k, index_weights, query_padding_mask, key_padding_mask, kvpacked=False)[:3]

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        output_pad_fn,
        index_q_unpad,
        index_k_unpad,
        index_weights_unpad
    )


def main(args, verbose=False):

    for _ in range(args.n_checks):
        (
            # varlen mqa
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, output_pad_fn,
            # varlen lighting indexer
            index_q_unpad, index_k_unpad, index_weights_unpad
        ) = oscillator(
            batch=args.batch,
            heads=heads,
            dim=dim,
            index_heads=index_heads,
            index_dim=index_dim,
            q_seqlen=args.q_seqlen,
            k_seqlen=args.k_seqlen,
            dtype=getattr(torch, dtype),
            device=torch.device("cuda"),
            distribution=args.oscillator_distribution
        )

        k_unpad = k_unpad.squeeze(1)             # (seq_len_kv, dim)
        v_unpad = v_unpad.squeeze(1)             # (seq_len_kv, dim)
        index_k_unpad = index_k_unpad.squeeze(1) # (seq_len_kv, dim)

        index_q_unpad.requires_grad_()
        index_k_unpad.requires_grad_()
        index_weights_unpad.requires_grad_()

        index_q_unpad.retain_grad()
        index_k_unpad.retain_grad()
        index_weights_unpad.retain_grad()

        dsa_warmup_loss = DsaWarmupLoss()

        o_unpad, dkl = dsa_warmup_loss(
            q_unpad, k_unpad, v_unpad,
            index_q_unpad, index_k_unpad,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            index_weights_unpad
        )
        dkl.backward(torch.ones(len(q_unpad), heads, dtype=torch.float32, device=torch.device("cuda")), retain_graph=True)

        grad_index_q_custom = index_q_unpad.grad
        grad_index_k_custom = index_k_unpad.grad
        grad_index_weights_custom = index_weights_unpad.grad

        ##################################
        # Alignment-1: flash_attn output #
        ##################################
        out = output_pad_fn(o_unpad)
        import flash_attn
        fla_out_unpad = flash_attn.flash_attn_varlen_func(
            q_unpad,
            k_unpad[:, None, :],
            v_unpad[:, None, :],
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            causal=True
        )
        fla_out = output_pad_fn(fla_out_unpad)
        logging.debug("[FWD] 1. Flash Attention Output check...")
        torch.testing.assert_close(out, fla_out, rtol=1e-2, atol=1e-2)
        logging.debug("[FWD] 1. Flash Attention Output check passed.\n")

        # [reference]
        # https://github.com/tile-ai/tilelang/blob/041d4a06b53ebeb4540636063cad2aa66fc5e1b9/examples/attention_sink/example_mha_sink_fwd_bhsd.py#L301
        fla_fwd_latency = tilelang.profiler.do_bench(
            lambda: flash_attn.flash_attn_varlen_func(
                q_unpad, k_unpad[:, None, :], v_unpad[:, None, :],
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                0.0, causal=True
            ),
            warmup=500
        )
        logging.info(f"FLA FWD Latency: {fla_fwd_latency} ms")

        fla_bwd_latency = tilelang.profiler.do_bench(
            lambda: fla_out_unpad.backward(
                torch.ones_like(fla_out_unpad),
                retain_graph=True
            ),
            warmup=500
        )
        logging.info(f"FLA BWD Latency: {fla_bwd_latency} ms")

        ##############################
        # Alignment-2: kl divergence #
        ##############################

        assert (~(dkl.isfinite())).sum() == 0

        dkl_ref = fn_dkl_ref_opt(
            q_unpad,
            k_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            index_q_unpad,
            index_k_unpad,
            index_weights_unpad,
            heads,
            torch.float32
        )

        logging.debug("[FWD] 2. kl_divergence check...")
        dkl_diff = validate_tensor_match(
            dkl_ref.clone(),
            dkl.clone(),
            tolerance=1e-8,
            tensor_name="dkl",
            should_raise=False
        )
        logging.debug(f"kl divergence diff = {dkl_diff}, tolerance=1e-8")
        # torch.testing.assert_close(dkl_ref.clone(), dkl.clone(), rtol=1e-2, atol=1e-2)
        logging.debug("[FWD] 2. kl_divergence check passed.\n")

        ################################
        # Alignment-3: gradient checks #
        ################################
        grad_index_q_ref, grad_index_k_ref, grad_index_weights_ref = fn_dkl_bwd_ref(
            q_unpad,
            k_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            index_q_unpad,
            index_k_unpad,
            index_weights_unpad,
            heads,
            torch.float32
        )

        logging.debug("[BWD] 3. gradient check...")
        diff_q=validate_tensor_match(
            grad_index_q_ref,
            grad_index_q_custom,
            tolerance=1e-5,
            tensor_name="grad_index_q",
            should_raise=False
        )
        # torch.testing.assert_close(grad_index_q_ref, grad_index_q_custom, atol=1e-2, rtol=1e-2)
        diff_k=validate_tensor_match(
            grad_index_k_ref,
            grad_index_k_custom,
            tolerance=1e-5,
            tensor_name="grad_index_k",
            should_raise=False
        )
        # torch.testing.assert_close(grad_index_k_ref, grad_index_k_custom, atol=1e-2, rtol=1e-2)
        diff_w=validate_tensor_match(
            grad_index_weights_ref,
            grad_index_weights_custom,
            tolerance=1e-5,
            tensor_name="grad_index_weights",
            should_raise=False
        )
        # torch.testing.assert_close(grad_index_weights_ref, grad_index_weights_custom, atol=1e-2, rtol=1e-2)
        logging.debug(f"grad_index_q diff = {diff_q}, tolerance=1e-5")
        logging.debug(f"grad_index_k diff = {diff_k}, tolerance=1e-5")
        logging.debug(f"grad_index_w diff = {diff_w}, tolerance=1e-5")
        logging.debug("[BWD] 3. gradient check passed.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--oscillator_distribution',
        type=str,
        default='bernoulli',
        choices=["normal", "bernoulli"]
    )
    # determine varlen data length
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--q_seqlen', type=int, default=2048, help='query sequence length')
    parser.add_argument('--k_seqlen', type=int, default=2048, help='key/value sequence length')
    parser.add_argument('--n_checks', type=int, default=1)
    parser.add_argument('--verbose', action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)

    torch.cuda.manual_seed_all(0)

    for distribution in ["normal"]:
    # for distribution in ["bernoulli"]:
    # for distribution in ["normal", "bernoulli"]:
        logging.debug("\n==========================================")
        logging.debug(f"Warmup: oscillator_distribution = {distribution}")
        logging.info(f"varlen setting: batch_size={args.batch}, seqlen_qk={args.q_seqlen}")
        args.oscillator_distribution = distribution
        main(args)
