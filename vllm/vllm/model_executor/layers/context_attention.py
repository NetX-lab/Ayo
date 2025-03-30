# SPDX-License-Identifier: Apache-2.0

# The kernels in this file are adapted from LightLLM's context_attention_fwd & vllm's prefix_prefill:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/prefix_prefill.py

import torch
import triton
import triton.language as tl
import platform
import os



if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability()
    is_ampere_or_later = device_capability[0] >= 8
    BASE_BLOCK = 128 if is_ampere_or_later else 64
    IS_TURING = device_capability == (7, 5)
else:
    BASE_BLOCK = 64
    IS_TURING = False


# Static kernels parameters
# BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
# NUM_WARPS = 4 if current_platform.is_rocm() else 8

IS_ROCM = 'rocm' in os.environ.get('TORCH_CUDA_ARCH_LIST', '').lower() or \
          'hip' in platform.platform().lower()
NUM_WARPS = 4 if IS_ROCM else 8


if triton.__version__ >= "2.1.0":

    @triton.jit
    def _fwd_kernel(
        Q,                      # Query tensor [batch_size*seq_len, num_heads, head_dim]
                                # Example: shape [32*128, 32, 128] for batch=32, seq=128, heads=32, dim=128
        K,                      # Key tensor for new tokens [batch_size*seq_len, num_kv_heads, head_dim]
                                # Example: shape [32*16, 32, 128] for batch=32, new_tokens=16, kv_heads=32
        V,                      # Value tensor for new tokens [batch_size*seq_len, num_kv_heads, head_dim]
                                # Example: shape [32*16, 32, 128]
        K_cache,                # Key cache tensor [num_blocks, num_kv_heads, head_dim/x, block_size, x]
                                # Example: shape [1024, 32, 16, 16, 8] for block_size=16, head_dim=128, x=8
        V_cache,                # Value cache tensor [num_blocks, num_kv_heads, head_dim, block_size]
                                # Example: shape [1024, 32, 128, 16]
        B_Loc,                  # Block location tensor [batch_size, max_blocks_per_seq]
                                # Maps sequence positions to block indices in KV cache
                                # Example: shape [32, 8] with values like [5, 8, 12, 0, 0, ...] 
        sm_scale,               # Softmax scale factor, typically 1/sqrt(head_dim)
                                # Example: 0.088 for head_dim=128
        k_scale,                # Scale factor for FP8 quantized K cache
                                # Example: tensor with single value like 0.1
        v_scale,                # Scale factor for FP8 quantized V cache
                                # Example: tensor with single value like 0.1
        B_Start_Loc,            # Batch start locations [batch_size+1]
                                # Stores starting position of each sequence in the global tensor
                                # Example: [0, 128, 256, 384] for 3 sequences of length 128
        B_Seqlen,               # Batch sequence lengths [batch_size]
                                # Total length of each sequence (context + query)
                                # Example: [1024, 2048, 512] for 3 sequences
        block_size,             # Size of blocks in KV cache
                                # Example: 16
        x,                      # Dimension size for K cache layout optimization
                                # Example: 8 (for head_dim=128, we have head_dim/x=16)
        Out,                    # Output tensor [batch_size*seq_len, num_heads, head_dim]
                                # Example: shape [32*128, 32, 128]
        
            # Stride parameters
        stride_b_loc_b,         # B_Loc stride for batch dimension
        stride_b_loc_s,         # B_Loc stride for sequence dimension
        stride_qbs,             # Q stride for batch*seq dimension
        stride_qh,              # Q stride for head dimension
        stride_qd,              # Q stride for head_dim dimension
        stride_kbs,             # K stride for batch*seq dimension
        stride_kh,              # K stride for head dimension
        stride_kd,              # K stride for head_dim dimension
        stride_vbs,             # V stride for batch*seq dimension
        stride_vh,              # V stride for head dimension
        stride_vd,              # V stride for head_dim dimension
        stride_obs,             # Out stride for batch*seq dimension
        stride_oh,              # Out stride for head dimension
        stride_od,              # Out stride for head_dim dimension
        stride_k_cache_bs,      # K_cache stride for block dimension
        stride_k_cache_h,       # K_cache stride for head dimension
        stride_k_cache_d,       # K_cache stride for head_dim/x dimension
        stride_k_cache_bl,      # K_cache stride for block_size dimension
        stride_k_cache_x,       # K_cache stride for x dimension
        stride_v_cache_bs,      # V_cache stride for block dimension
        stride_v_cache_h,       # V_cache stride for head dimension
        stride_v_cache_d,       # V_cache stride for head_dim dimension
        stride_v_cache_bl,      # V_cache stride for block_size dimension
        
        # Configuration parameters
        num_queries_per_kv: int,  # Number of query heads per KV head (for grouped attention)
                                # Example: 1 for standard attention, 2+ for grouped attention
        IN_PRECISION: tl.constexpr,  # Precision for matrix multiplication
                                    # Example: None (auto) or 'ieee' for float32
        BLOCK_M: tl.constexpr,    # Block size for query dimension
                                # Example: 64 or 128
        BLOCK_DMODEL: tl.constexpr,  # Head size/dimension
                                    # Example: 128
        BLOCK_DMODEL_PADDED: tl.constexpr,  # Head size padded to power of 2
                                            # Example: 128 (if already power of 2) or 256
        BLOCK_N: tl.constexpr,    # Block size for KV dimension
                                # Example: 64 or 128
        SLIDING_WINDOW: tl.constexpr,  # Size of sliding window for attention
                                        # Example: 0 (disabled) or 4096 (limit to 4K tokens)
        SKIP_DECODE: tl.constexpr,  # Whether to skip single-token decode mode
                                    # Example: False (process all) or True (skip length-1 queries)
    ):

        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                               cur_batch_in_all_start_index)
        cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

        if SKIP_DECODE and cur_batch_query_len == 1:
            return

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                       dtype=tl.float32)  # [M,D]

        # compute query against context (no causal mask here)
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)  # [N]
            # [D,N]
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            # [N,D]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)

            k_load = tl.load(K_cache + off_k,
                             mask=dim_mask[:, None] &
                             ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                             other=0.0)  # [D,N]

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                              (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                              -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)  # [M]
            p = tl.exp(qk - m_ij[:, None])  # [M,N]
            l_ij = tl.sum(p, 1)  # [M]
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)  # [M]
            alpha = tl.exp(m_i - m_i_new)  # [M]
            beta = tl.exp(m_ij - m_i_new)  # [M]
            l_i_new = alpha * l_i + beta * l_ij  # [M]

            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)  # [N,D]
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk *= sm_scale
            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] - (start_n + offs_n[None, :])
                    < SLIDING_WINDOW, qk, -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_query_len))
        return

    @triton.jit
    def _fwd_kernel_flash_attn_v2(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        q = tl.load(Q + off_q,
                    mask=offs_m[:, None]
                    < cur_batch_seq_len - cur_batch_ctx_len,
                    other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            k = tl.load(K_cache + off_k,
                        mask=(start_n + offs_n[None, :]) < cur_batch_ctx_len,
                        other=0.0)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(V_cache + off_v,
                        mask=(start_n + offs_n[:, None]) < cur_batch_ctx_len,
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(
            block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=(start_n + offs_n[None, :])
                        < cur_batch_seq_len - cur_batch_ctx_len,
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=(start_n + offs_n[:, None])
                        < cur_batch_seq_len - cur_batch_ctx_len,
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        # acc /= l_i[:, None]
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len)
        return

    @triton.jit
    def _fwd_kernel_alibi(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        Alibi_slopes,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        IN_PRECISION: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
        SKIP_DECODE: tl.constexpr,
    ):
        # attn_bias[]
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        # cur_batch_seq_len: the length of prompts
        # cur_batch_ctx_len: the length of prefix
        # cur_batch_in_all_start_index: the start id of the dim=0
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                               cur_batch_in_all_start_index)
        cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

        if SKIP_DECODE and cur_batch_query_len == 1:
            return

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
                    other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = 0
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----

            # the block location of the KV cache, [num_blocks,...]
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            # the offset of the K cache, [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
                
            # the offset of the V cache, [num_blocks, num_kv_heads, head_dim, block_size]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            
            k_load = tl.load(K_cache + off_k,
                             mask=dim_mask[:, None] &
                             ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                             other=0.0)  # [D,N]

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale

            # load alibi
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision='ieee')
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(
            block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        # init alibi
        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = cur_batch_ctx_len
        # # init debugger
        # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
        # offset_db_k = tl.arange(0, BLOCK_N)
        # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :])
                         < cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk, input_precision='ieee')
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))

            # load alibi
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None])
                         < cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision='ieee')
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        acc = acc / l_i[:, None]

        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len))
        return

    @torch.inference_mode()
    def context_attention_fwd(q,
                              k,
                              v,
                              o,
                              kv_cache_dtype: str,
                              k_cache,
                              v_cache,
                              b_loc,
                              b_start_loc,
                              b_seq_len,
                              max_input_len,
                              k_scale: torch.Tensor,
                              v_scale: torch.Tensor,
                              alibi_slopes=None,
                              sliding_window=None,
                              sm_scale=None,
                              skip_decode=False):

        q_dtype_is_f32 = q.dtype is torch.float32
        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        # if q.dtype is torch.float32:
        BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK

        # Turing does have tensor core for float32 multiplication
        # use ieee as fallback for triton kernels work. There is also
        # warning on vllm/config.py to inform users this fallback
        # implementation
        IN_PRECISION = 'ieee' if IS_TURING and q_dtype_is_f32 else None

        # Conversion of FP8 Tensor from uint8 storage to
        # appropriate torch.dtype for interpretation by Triton
        if "fp8" in kv_cache_dtype:
            assert (k_cache.dtype == torch.uint8)
            assert (v_cache.dtype == torch.uint8)

            if kv_cache_dtype in ("fp8", "fp8_e4m3"):
                target_dtype = torch.float8_e4m3fn
            elif kv_cache_dtype == "fp8_e5m2":
                target_dtype = torch.float8_e5m2
            else:
                raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        if k_cache is not None and (k_cache.dtype == torch.uint8
                or v_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
            raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache prefill kernel")

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # round up Lk to a power of 2 - this is required for Triton block size
        Lk_padded = triton.next_power_of_2(Lk)

        if sm_scale is None:
            sm_scale = 1.0 / (Lq**0.5)
        batch, head = b_seq_len.shape[0], q.shape[1]
        num_queries_per_kv = q.shape[1] // k.shape[1]

        assert batch + 1 == len(b_start_loc)
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

        # 0 means "disable"
        if sliding_window is None or sliding_window <= 0:
            sliding_window = 0

        if alibi_slopes is not None:
            _fwd_kernel_alibi[grid](
                q,
                k,
                v,
                k_cache,
                v_cache,
                b_loc,
                sm_scale,
                k_scale,
                v_scale,
                b_start_loc,
                b_seq_len,
                alibi_slopes,
                v_cache.shape[3],
                k_cache.shape[4],
                o,
                b_loc.stride(0),
                b_loc.stride(1),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                k_cache.stride(
                    4
                ),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(
                    3),  #[num_blocks, num_kv_heads, head_size, block_size]
                num_queries_per_kv=num_queries_per_kv,
                IN_PRECISION=IN_PRECISION,
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_DMODEL_PADDED=Lk_padded,
                BLOCK_N=BLOCK,
                SKIP_DECODE=skip_decode,
                num_warps=NUM_WARPS,
                num_stages=1,
            )
            return

        _fwd_kernel[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(
                4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SLIDING_WINDOW=sliding_window,
            SKIP_DECODE=skip_decode,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return