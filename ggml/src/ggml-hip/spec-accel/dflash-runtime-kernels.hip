// SPDX-License-Identifier: MIT
// BeeLlama lifecycle/ring-cache adaptations for the BridgeSpec DFlash2 kernels.

__global__ void df_rope_mrope4(
        float * __restrict__ x,
        int nh,
        const int32_t * __restrict__ positions,
        int ncols,
        int section0,
        int section1,
        int section2,
        float frequency_base) {
    const int head = blockIdx.x;
    const int col = blockIdx.y;
    const int pair = threadIdx.x;
    if (pair >= DF_HDIM / 2) {
        return;
    }
    const int boundary1 = section0;
    const int boundary2 = boundary1 + section1;
    const int boundary3 = boundary2 + section2;
    const int plane = pair < boundary1 ? 0 : (pair < boundary2 ? 1 : (pair < boundary3 ? 2 : 3));
    float * row = x + ((size_t) col * nh + head) * DF_HDIM;
    const float theta = positions[(size_t) plane * ncols + col] *
            powf(frequency_base, -2.0f * pair / DF_HDIM);
    const float cosine = cosf(theta);
    const float sine = sinf(theta);
    const float x0 = row[pair];
    const float x1 = row[pair + DF_HDIM / 2];
    row[pair] = x0 * cosine - x1 * sine;
    row[pair + DF_HDIM / 2] = x0 * sine + x1 * cosine;
}

__global__ void df_kv_store_ring(
        const float * __restrict__ k,
        const float * __restrict__ v,
        __half * __restrict__ k_cache,
        __half * __restrict__ v_cache,
        const int32_t * __restrict__ positions,
        int ncols) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    constexpr int width = DF_KV * DF_HDIM;
    if (element >= width || col >= ncols) {
        return;
    }
    const int position = positions[col];
    const size_t destination = (size_t) (position % DF_WINDOW) * width + element;
    k_cache[destination] = __float2half(k[(size_t) col * width + element]);
    v_cache[destination] = __float2half(v[(size_t) col * width + element]);
}

// Non-causal eight-row DFlash attention over a resident 2048-token ring.  The
// caller proves that every position in [max(0,qpos-2047), n_past) is resident.
__global__ void df_attn_ring(
        const float * __restrict__ q,
        const __half * __restrict__ k_cache,
        const __half * __restrict__ v_cache,
        const float * __restrict__ block_k,
        const float * __restrict__ block_v,
        float * __restrict__ output,
        const int32_t * __restrict__ query_positions) {
    const int n_past = query_positions[0];
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    const int kv_head = head / (DF_HEADS / DF_KV);
    __shared__ float reduction[256];
    __shared__ float block_scores[DF_BLOCK];
    const float * query = q + ((size_t) row * DF_HEADS + head) * DF_HDIM;
    const float scale = rsqrtf((float) DF_HDIM);
    const int query_pos = query_positions[row];
    const int lower = max(0, query_pos - (DF_WINDOW - 1));

    float maximum = -1e30f;
    float sum = 0.0f;
    float accumulator = 0.0f;
    if (lane < DF_BLOCK) {
        const float * key = block_k + ((size_t) lane * DF_KV + kv_head) * DF_HDIM;
        float score = 0.0f;
        for (int i = 0; i < DF_HDIM; ++i) {
            score += query[i] * key[i];
        }
        block_scores[lane] = score * scale;
    }
    __syncthreads();
    for (int i = 0; i < DF_BLOCK; ++i) {
        maximum = fmaxf(maximum, block_scores[i]);
    }

    constexpr int chunk = 1024;
    __shared__ float scores[chunk];
    for (int begin = lower; begin < n_past; begin += chunk) {
        const int count = min(chunk, n_past - begin);
        for (int offset = lane; offset < count; offset += blockDim.x) {
            const int position = begin + offset;
            const __half * key = k_cache +
                    ((size_t) (position % DF_WINDOW) * DF_KV + kv_head) * DF_HDIM;
            float score = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < DF_HDIM; ++i) {
                score += query[i] * __half2float(key[i]);
            }
            scores[offset] = score * scale;
        }
        __syncthreads();
        float chunk_maximum = -1e30f;
        for (int offset = lane; offset < count; offset += blockDim.x) {
            chunk_maximum = fmaxf(chunk_maximum, scores[offset]);
        }
        reduction[lane] = chunk_maximum;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) {
                reduction[lane] = fmaxf(reduction[lane], reduction[lane + offset]);
            }
            __syncthreads();
        }
        chunk_maximum = reduction[0];
        const float new_maximum = fmaxf(maximum, chunk_maximum);
        float chunk_sum = 0.0f;
        for (int offset = lane; offset < count; offset += blockDim.x) {
            const float weight = expf(scores[offset] - new_maximum);
            scores[offset] = weight;
            chunk_sum += weight;
        }
        reduction[lane] = chunk_sum;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) {
                reduction[lane] += reduction[lane + offset];
            }
            __syncthreads();
        }
        const float old_scale = expf(maximum - new_maximum);
        if (lane < DF_HDIM) {
            float value_sum = 0.0f;
            for (int offset = 0; offset < count; ++offset) {
                const int position = begin + offset;
                const size_t value_index =
                        ((size_t) (position % DF_WINDOW) * DF_KV + kv_head) * DF_HDIM + lane;
                value_sum += scores[offset] * __half2float(v_cache[value_index]);
            }
            accumulator = accumulator * old_scale + value_sum;
        }
        sum = sum * old_scale + reduction[0];
        maximum = new_maximum;
        __syncthreads();
    }

    __shared__ float block_weights[DF_BLOCK];
    if (lane == 0) {
        for (int i = 0; i < DF_BLOCK; ++i) {
            block_weights[i] = expf(block_scores[i] - maximum);
        }
    }
    __syncthreads();
    float block_sum = 0.0f;
    for (int i = 0; i < DF_BLOCK; ++i) {
        block_sum += block_weights[i];
    }
    if (lane < DF_HDIM) {
        float value_sum = 0.0f;
        for (int i = 0; i < DF_BLOCK; ++i) {
            value_sum += block_weights[i] *
                    block_v[((size_t) i * DF_KV + kv_head) * DF_HDIM + lane];
        }
        accumulator += value_sum;
        output[((size_t) row * DF_HEADS + head) * DF_HDIM + lane] =
                accumulator / (sum + block_sum);
    }
}

__global__ void df_shift_k_ring(
        __half * __restrict__ cache,
        int32_t begin,
        int32_t count,
        int32_t delta,
        float frequency_base) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int pair = threadIdx.x;
    if (row >= count || head >= DF_KV || pair >= DF_HDIM / 2) {
        return;
    }
    const int position = begin + row;
    __half * key = cache + ((size_t) (position % DF_WINDOW) * DF_KV + head) * DF_HDIM;
    const float theta = delta * powf(frequency_base, -2.0f * pair / DF_HDIM);
    const float cosine = cosf(theta);
    const float sine = sinf(theta);
    const float x0 = __half2float(key[pair]);
    const float x1 = __half2float(key[pair + DF_HDIM / 2]);
    key[pair] = __float2half(x0 * cosine - x1 * sine);
    key[pair + DF_HDIM / 2] = __float2half(x0 * sine + x1 * cosine);
}
// Row-major materialization used only by the large-row feature-injection
// route.  The quantized tensors remain authoritative for the captured
// eight-row decode graph.
__global__ void df_q4k_matrix_to_f16(
        const block_q4_K * __restrict__ weights,
        __half * __restrict__ output,
        size_t elements,
        int width) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int column = int(index % width);
    const size_t row = index / width;
    const block_q4_K * block = weights + row * (width / 256) + column / 256;
    const int within = column & 255;
    const int group = within / 64;
    const int high = (within & 63) >= 32;
    float scale;
    float minimum;
    q4k_scale_min(group * 2 + high, block->scales, &scale, &minimum);
    const uint8_t packed = block->qs[group * 32 + (within & 31)];
    const int quant = high ? packed >> 4 : packed & 0x0f;
    output[index] = __float2half(
            df_f16(block->d) * scale * quant - df_f16(block->dmin) * minimum);
}

__global__ void df_q6k_matrix_to_f16(
        const block_q6_K * __restrict__ weights,
        __half * __restrict__ output,
        size_t elements,
        int width) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int column = int(index % width);
    const size_t row = index / width;
    const block_q6_K * block = weights + row * (width / 256) + column / 256;
    const int within = column & 255;
    const int half = within / 128;
    const int local = within & 127;
    const int lane = local & 31;
    const uint8_t low = block->ql[half * 64 + (local & 63)];
    const uint8_t high = block->qh[half * 32 + lane];
    const int shift = (local / 32) * 2;
    const int quant = int(((local >= 64 ? low >> 4 : low & 0x0f) |
            (((high >> shift) & 0x03) << 4))) - 32;
    const int scale = block->scales[half * 8 + local / 16];
    output[index] = __float2half(df_f16(block->d) * scale * quant);
}
