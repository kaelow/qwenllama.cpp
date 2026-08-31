// SPDX-License-Identifier: MIT
//
// BeeLlama opt-in speculative accelerator for ROCm/HIP.  The MTP kernels are
// adapted from BridgeSpec 0.1.0 (see licenses/BridgeSpec-MIT.txt).  The
// lifecycle, per-sequence state, M-RoPE handling, checkpoints, and backend ABI
// are BeeLlama implementations; no external sidecar DLL is loaded.

#include "ggml-spec-accel.h"
#include "gguf.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bridge-mtp-kernels.hip"
#include "bridge-mtp-kernels-batch.hip"
#include "bridge-dflash-kernels.hip"
#include "dflash-runtime-kernels.hip"

namespace {

thread_local std::string global_error;

constexpr uint32_t SPEC_ARTIFACT_VERSION = 1;
constexpr uint32_t SPEC_STATE_VERSION    = 3;
constexpr uint32_t SPEC_STATE_MAGIC      = 0x53414342; // BCAS
constexpr int32_t  SPEC_VOCAB            = 248320;
constexpr int32_t  SPEC_MTP_BLOCK        = 64;
constexpr int32_t  SPEC_CATCHUP_TILE     = 8;

struct tensor_allocation {
    std::string name;
    void * device = nullptr;
    void * host = nullptr;
    size_t size = 0;
};

struct mtp_weights {
    const block_q4_0 * embd = nullptr;
    const block_q4_0 * ehp  = nullptr;
    const block_q4_0 * q    = nullptr;
    const block_q4_0 * k    = nullptr;
    const block_q4_0 * v    = nullptr;
    const block_q4_0 * o    = nullptr;
    const block_q4_0 * fg   = nullptr;
    const block_q4_0 * fu   = nullptr;
    const block_q4_0 * fd   = nullptr;
    const block_q4_0 * head = nullptr;

    const float * enorm = nullptr;
    const float * hnorm = nullptr;
    const float * attn  = nullptr;
    const float * qnorm = nullptr;
    const float * knorm = nullptr;
    const float * post  = nullptr;
    const float * shnorm = nullptr;
};

struct dflash_layer_weights {
    const block_q4_K * conv_a_proj = nullptr;
    const block_q4_K * conv_f_proj = nullptr;
    const block_q4_K * wq = nullptr;
    const block_q4_K * wk = nullptr;
    const block_q4_K * wo = nullptr;
    const block_q4_K * up = nullptr;
    const block_q4_K * gate = nullptr;
    const void * wv = nullptr;
    const void * down = nullptr;
    bool wv_q6 = false;
    bool down_q6 = false;
    const float * attn_norm = nullptr;
    const float * qnorm = nullptr;
    const float * knorm = nullptr;
    const float * ffn_norm = nullptr;
    const float * conv_a_base = nullptr;
    const float * conv_f_base = nullptr;
};

struct dflash_weights {
    dflash_layer_weights layers[DF_LAYERS];
    const block_q4_K * fc = nullptr;
    const block_q4_K * selector_hidden = nullptr;
    const block_q4_0d * target_embedding = nullptr;
    const block_q6_K * target_head = nullptr;
    const float * enc_norm = nullptr;
    const float * output_norm = nullptr;
    std::vector<float> selector_prev;
    std::vector<float> selector_next;
    std::vector<int32_t> remap;
    std::array<int32_t, 4> mrope_sections { 64, 0, 0, 0 };
    std::array<int32_t, 5> target_layers {};
    float rope_freq_base = DF_FBASE;
    int32_t mask_token_id = -1;
};

struct spec_runtime;

struct mtp_scratch {
    float * x_tok = nullptr;
    float * x_h = nullptr;
    float * e_n = nullptr;
    float * h_n = nullptr;
    float * cat = nullptr;
    float * proj = nullptr;
    float * attin = nullptr;
    float * qfull = nullptr;
    float * q = nullptr;
    float * gate = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * attout = nullptr;
    float * ffn_in = nullptr;
    float * ffn_g = nullptr;
    float * ffn_u = nullptr;
    float * ffn_tmp = nullptr;
    float * resid = nullptr;
    float * hnorm2 = nullptr;
    int32_t * token = nullptr;
    int32_t * pos4 = nullptr;
};

struct mtp_batch_scratch {
    uint32_t capacity = 0;
    float * x_tok = nullptr;
    float * h = nullptr;
    float * e_n = nullptr;
    float * h_n = nullptr;
    float * cat = nullptr;
    float * proj = nullptr;
    float * attin = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    int32_t * tokens = nullptr;
    int32_t * pos4 = nullptr;
    int32_t * cache_rows = nullptr;
    __half * gemm_input = nullptr;
};

struct mtp_dense_weights {
    __half * ehp = nullptr;
    __half * k = nullptr;
    __half * v = nullptr;
    bool ready = false;
    bool unavailable = false;
};

// The DFlash decode graph keeps its fixed eight-row buffers below.  Feature
// injection can receive a full target ubatch, so its batched storage is kept
// separate and sized from the actual row count and currently free VRAM.
struct dflash_batch_scratch {
    uint32_t capacity = 0;
    float * features = nullptr;
    float * encoded = nullptr;
    float * xn = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    int32_t * positions = nullptr;
    __half * gemm_input = nullptr;
};

struct dflash_dense_weights {
    __half * fc = nullptr;
    __half * k[DF_LAYERS] {};
    __half * v[DF_LAYERS] {};
    bool ready = false;
    bool unavailable = false;
};

struct dflash_scratch {
    float * features = nullptr;
    float * encoded = nullptr;
    float * x = nullptr;
    float * xn = nullptr;
    float * dyn_a = nullptr;
    float * dyn_f = nullptr;
    float * conv = nullptr;
    float * q = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * attention = nullptr;
    float * output = nullptr;
    float * ffn_up = nullptr;
    float * ffn_gate = nullptr;
    float * ffn_tmp = nullptr;
    float * token_embedding = nullptr;
    float * logits = nullptr;
    float * selector_hidden = nullptr;
    float * topk_part_max = nullptr;
    float * topk_values = nullptr;
    int32_t * positions = nullptr;
    int32_t * tokens = nullptr;
    int32_t * topk_part_ids = nullptr;
    int32_t * topk_ids = nullptr;
    __half * k_cache[DF_LAYERS] {};
    __half * v_cache[DF_LAYERS] {};
    uint32_t feature_capacity = 0;
    int32_t topk_parts = 0;
    hipGraphExec_t graph_exec = nullptr;
};

struct prepared_durable_state {
    void * k[DF_LAYERS] {};
    void * v[DF_LAYERS] {};
    size_t mtp_capacity = 0;
    size_t input_size = 0;
    uint64_t input_checksum = 0;
    uint64_t source_generation = 0;
    uint64_t saved_generation = 0;
    int32_t logical_pos = 0;
    int32_t coverage_begin = 0;
    int32_t coverage_end = 0;
    uint32_t flags = 0;
    std::vector<float> pending_hidden;
    std::vector<std::array<int32_t, 4>> mtp_positions;
    bool ready = false;
};

struct spec_sequence {
    spec_runtime * runtime = nullptr;
    int32_t id = -1;
    hipStream_t stream = nullptr;
    hipblasHandle_t blas = nullptr;

    mtp_scratch scratch;
    mtp_batch_scratch batch;
    dflash_scratch dflash;
    dflash_batch_scratch dflash_batch;
    prepared_durable_state prepared;

    void * k_cache = nullptr;
    void * v_cache = nullptr;
    size_t cache_capacity = 0;

    float * part_m = nullptr;
    float * part_l = nullptr;
    float * part_acc = nullptr;
    float * logits = nullptr;
    float * pmax = nullptr;
    int32_t * pidx = nullptr;
    int32_t * token_out = nullptr;
    int32_t n_parts = 0;

    int32_t logical_pos = 0;
    int32_t coverage_begin = 0;
    int32_t coverage_end = 0;
    uint64_t generation = 1;
    std::vector<float> pending_hidden;
    // MTP cache storage is indexed by input record, not by temporal model
    // position.  M-RoPE images can have hundreds of distinct K/V records that
    // share one temporal position; retaining all four planes also makes suffix
    // truncation and durable restore unambiguous.
    std::vector<std::array<int32_t, 4>> mtp_positions;
    bool has_multimodal_positions = false;
    bool healthy = true;
};

struct spec_runtime {
    ggml_spec_accel_descriptor descriptor {};
    std::string artifact_path;
    std::string target_path;
    std::string model_fingerprint;
    std::string device_name;
    std::string error;

    int device = -1;
    hipStream_t upload_stream = nullptr;
    bool async_alloc = false;

    std::vector<tensor_allocation> allocations;
    std::unordered_map<std::string, void *> tensors;
    mtp_weights weights;
    mtp_dense_weights dense_weights;
    dflash_weights dflash;
    dflash_dense_weights dflash_dense;
    int32_t * remap = nullptr;

    std::mutex mutex;
    std::mutex dense_mutex;
    std::unordered_map<int32_t, spec_sequence *> sequences;
};

struct checkpoint_transaction {
    spec_sequence * sequence = nullptr;
    ggml_spec_accel_checkpoint_desc descriptor {};
    std::vector<float> pending_hidden;
};

struct state_header {
    uint32_t magic;
    uint32_t version;
    uint32_t kind;
    uint32_t cache_type;
    uint32_t flags;
    uint32_t reserved;
    int32_t logical_pos;
    int32_t coverage_begin;
    int32_t coverage_end;
    uint32_t record_count;
    uint32_t pending_count;
    uint64_t position_bytes;
    uint64_t generation;
    uint64_t cache_bytes;
    uint64_t checksum;
};

constexpr uint32_t SPEC_STATE_FLAG_MULTIMODAL_POSITIONS = 1u << 0;

static size_t cache_row_bytes(uint32_t cache_type) {
    const size_t elements = N_HEAD_KV * HEAD_DIM;
    if (cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0) {
        return elements / 32 * sizeof(spec_block_q8_0);
    }
    return elements * sizeof(__half);
}

static size_t cache_bytes(const spec_runtime * runtime, size_t positions) {
    return positions * cache_row_bytes(runtime->descriptor.cache_type);
}

static uint8_t * cache_at(void * cache, const spec_runtime * runtime, size_t position) {
    return static_cast<uint8_t *>(cache) + cache_bytes(runtime, position);
}

static const uint8_t * cache_at(const void * cache, const spec_runtime * runtime, size_t position) {
    return static_cast<const uint8_t *>(cache) + cache_bytes(runtime, position);
}

static uint64_t fnv1a(uint64_t value, const void * data, size_t size) {
    const auto * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        value = (value ^ p[i]) * 1099511628211ULL;
    }
    return value;
}

static bool valid_struct(const void * ptr, uint32_t size, uint32_t expected) {
    return ptr != nullptr && size >= expected;
}

static int32_t fail(spec_runtime * runtime, int32_t status, const std::string & message) {
    global_error = message;
    if (runtime != nullptr) {
        runtime->error = message;
    }
    return status;
}

static bool hip_ok(spec_runtime * runtime, hipError_t result, const char * operation) {
    if (result == hipSuccess) {
        return true;
    }
    global_error = std::string(operation) + ": " + hipGetErrorString(result);
    if (runtime != nullptr) {
        runtime->error = global_error;
    }
    return false;
}

static bool hipblas_ok(spec_runtime * runtime, hipblasStatus_t result, const char * operation) {
    if (result == HIPBLAS_STATUS_SUCCESS) {
        return true;
    }
    global_error = std::string(operation) + ": hipBLAS status " + std::to_string(int(result));
    if (runtime != nullptr) {
        runtime->error = global_error;
    }
    return false;
}

static void * device_alloc(spec_runtime * runtime, size_t size, hipStream_t stream) {
    void * result = nullptr;
    if (size == 0) {
        return nullptr;
    }
    if (runtime->async_alloc) {
        const hipError_t err = hipMallocAsync(&result, size, stream);
        if (err == hipSuccess) {
            return result;
        }
        // A runtime can expose the pool API but reject it for a particular
        // allocation.  Clear the sticky error and permanently use the safe
        // allocator for this runtime.
        hipGetLastError();
        runtime->async_alloc = false;
    }
    if (!hip_ok(runtime, hipMalloc(&result, size), "hipMalloc")) {
        return nullptr;
    }
    return result;
}

static void device_free(spec_runtime * runtime, void * ptr, hipStream_t stream) {
    if (ptr == nullptr) {
        return;
    }
    if (runtime != nullptr && runtime->async_alloc && stream != nullptr) {
        if (hipFreeAsync(ptr, stream) == hipSuccess) {
            return;
        }
        hipGetLastError();
    }
    hipFree(ptr);
}

static void clear_prepared_state(spec_sequence * sequence) {
    if (sequence == nullptr) {
        return;
    }
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        device_free(sequence->runtime, sequence->prepared.k[layer], sequence->stream);
        device_free(sequence->runtime, sequence->prepared.v[layer], sequence->stream);
    }
    sequence->prepared = {};
}

static int64_t key_id(const gguf_context * gguf, const char * key, gguf_type type) {
    const int64_t id = gguf_find_key(gguf, key);
    return id >= 0 && gguf_get_kv_type(gguf, id) == type ? id : -1;
}

static bool read_exact(FILE * file, uint64_t offset, void * output, size_t size) {
#if defined(_WIN32)
    if (_fseeki64(file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
        return false;
    }
#else
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        return false;
    }
#endif
    return size == 0 || std::fread(output, 1, size, file) == size;
}

static std::string tensor_name(int block, const char * suffix) {
    return "blk." + std::to_string(block) + "." + suffix;
}

struct expected_tensor {
    std::string name;
    ggml_type type;
    int64_t ne0;
    int64_t ne1;
};

static std::vector<expected_tensor> mtp_schema(int block) {
    return {
        {"token_embd.weight",                         GGML_TYPE_Q4_0,  5120, 248320},
        {tensor_name(block, "attn_k.weight"),         GGML_TYPE_Q4_0,  5120,   1024},
        {tensor_name(block, "attn_k_norm.weight"),    GGML_TYPE_F32,    256,      1},
        {tensor_name(block, "attn_norm.weight"),      GGML_TYPE_F32,   5120,      1},
        {tensor_name(block, "attn_output.weight"),    GGML_TYPE_Q4_0,  6144,   5120},
        {tensor_name(block, "attn_q.weight"),         GGML_TYPE_Q4_0,  5120,  12288},
        {tensor_name(block, "attn_q_norm.weight"),    GGML_TYPE_F32,    256,      1},
        {tensor_name(block, "attn_v.weight"),         GGML_TYPE_Q4_0,  5120,   1024},
        {tensor_name(block, "ffn_down.weight"),       GGML_TYPE_Q4_0, 17408,   5120},
        {tensor_name(block, "ffn_gate.weight"),       GGML_TYPE_Q4_0,  5120,  17408},
        {tensor_name(block, "ffn_up.weight"),         GGML_TYPE_Q4_0,  5120,  17408},
        {tensor_name(block, "nextn.eh_proj.weight"),  GGML_TYPE_Q4_0, 10240,   5120},
        {tensor_name(block, "nextn.enorm.weight"),    GGML_TYPE_F32,   5120,      1},
        {tensor_name(block, "nextn.hnorm.weight"),    GGML_TYPE_F32,   5120,      1},
        {tensor_name(block, "nextn.shared_head_norm.weight"), GGML_TYPE_F32, 5120, 1},
        {tensor_name(block, "post_attention_norm.weight"), GGML_TYPE_F32, 5120, 1},
        {tensor_name(block, "nextn.shared_head_head.weight"), GGML_TYPE_Q4_0, 5120, 40960},
    };
}

static std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016" PRIx64, value);
    return text;
}

static bool sample_target_fingerprint(const char * path, std::string & result, std::string & error) {
    result.clear();
    if (path == nullptr || path[0] == '\0') {
        error = "the accelerator requires the target model path for identity validation";
        return false;
    }

    gguf_init_params params {};
    params.no_alloc = true;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_file(path, params), gguf_free);
    if (!gguf) {
        error = std::string("cannot read target GGUF: ") + path;
        return false;
    }
    FILE * file = std::fopen(path, "rb");
    if (file == nullptr) {
        error = std::string("cannot open target GGUF: ") + std::strerror(errno);
        return false;
    }

    std::vector<std::string> names = {
        "token_embd.weight",
        "output.weight",
        tensor_name(SPEC_MTP_BLOCK, "attn_k.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_k_norm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_norm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_output.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_q.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_q_norm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "attn_v.weight"),
        tensor_name(SPEC_MTP_BLOCK, "ffn_down.weight"),
        tensor_name(SPEC_MTP_BLOCK, "ffn_gate.weight"),
        tensor_name(SPEC_MTP_BLOCK, "ffn_up.weight"),
        tensor_name(SPEC_MTP_BLOCK, "nextn.eh_proj.weight"),
        tensor_name(SPEC_MTP_BLOCK, "nextn.enorm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "nextn.hnorm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "nextn.shared_head_norm.weight"),
        tensor_name(SPEC_MTP_BLOCK, "post_attention_norm.weight"),
    };
    std::sort(names.begin(), names.end());

    uint64_t hash = 1469598103934665603ULL;
    std::array<uint8_t, 64> sample {};
    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    for (const std::string & name : names) {
        const int64_t id = gguf_find_tensor(gguf.get(), name.c_str());
        if (id < 0) {
            std::fclose(file);
            error = "target GGUF is missing identity tensor: " + name;
            return false;
        }
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        const uint32_t type = uint32_t(gguf_get_tensor_type(gguf.get(), id));
        const uint64_t size = gguf_get_tensor_size(gguf.get(), id);
        const uint64_t offset = data_offset + gguf_get_tensor_offset(gguf.get(), id);
        hash = fnv1a(hash, name.data(), name.size());
        hash = fnv1a(hash, &type, sizeof(type));
        hash = fnv1a(hash, ne, sizeof(int64_t) * GGML_MAX_DIMS);
        hash = fnv1a(hash, &size, sizeof(size));

        const std::array<uint64_t, 3> starts = {
            0,
            size > sample.size() ? (size - sample.size()) / 2 : 0,
            size > sample.size() ? size - sample.size() : 0,
        };
        for (uint64_t start : starts) {
            const size_t count = size_t(std::min<uint64_t>(sample.size(), size - start));
            if (!read_exact(file, offset + start, sample.data(), count)) {
                std::fclose(file);
                error = "failed to sample target identity tensor: " + name;
                return false;
            }
            hash = fnv1a(hash, sample.data(), count);
        }
    }
    std::fclose(file);
    result = hex64(hash);
    return true;
}

static void free_tensor_allocations(spec_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }
    for (auto it = runtime->allocations.rbegin(); it != runtime->allocations.rend(); ++it) {
        if (it->host != nullptr) {
            hipHostFree(it->host);
        } else {
            device_free(runtime, it->device, runtime->upload_stream);
        }
    }
    runtime->allocations.clear();
    runtime->tensors.clear();
    runtime->dense_weights = {};
    runtime->dflash_dense = {};
    device_free(runtime, runtime->remap, runtime->upload_stream);
    runtime->remap = nullptr;
    if (runtime->upload_stream != nullptr) {
        hipStreamSynchronize(runtime->upload_stream);
    }
}

static bool upload_tensor(
        spec_runtime * runtime,
        FILE * file,
        const gguf_context * gguf,
        int64_t id,
        bool host_mapped) {
    const char * name = gguf_get_tensor_name(gguf, id);
    const size_t size = gguf_get_tensor_size(gguf, id);
    const uint64_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id);

    tensor_allocation allocation;
    allocation.name = name;
    allocation.size = size;

    if (host_mapped) {
        void * host = nullptr;
        if (!hip_ok(runtime, hipHostMalloc(&host, size, hipHostMallocMapped), "hipHostMalloc(mapped embedding)")) {
            return false;
        }
        allocation.host = host;
        if (!read_exact(file, offset, host, size)) {
            hipHostFree(host);
            runtime->error = "short read while loading " + allocation.name;
            return false;
        }
        if (!hip_ok(runtime, hipHostGetDevicePointer(&allocation.device, host, 0), "hipHostGetDevicePointer")) {
            hipHostFree(host);
            return false;
        }
    } else {
        std::vector<uint8_t> data;
        try {
            data.resize(size);
        } catch (const std::bad_alloc &) {
            runtime->error = "host allocation failed while loading " + allocation.name;
            return false;
        }
        if (!read_exact(file, offset, data.data(), size)) {
            runtime->error = "short read while loading " + allocation.name;
            return false;
        }
        allocation.device = device_alloc(runtime, size, runtime->upload_stream);
        if (allocation.device == nullptr) {
            return false;
        }
        if (!hip_ok(runtime, hipMemcpyAsync(allocation.device, data.data(), size,
                        hipMemcpyHostToDevice, runtime->upload_stream), "upload accelerator tensor")) {
            device_free(runtime, allocation.device, runtime->upload_stream);
            return false;
        }
        if (!hip_ok(runtime, hipStreamSynchronize(runtime->upload_stream), "synchronize accelerator tensor upload")) {
            device_free(runtime, allocation.device, runtime->upload_stream);
            return false;
        }
    }

    runtime->tensors.emplace(allocation.name, allocation.device);
    runtime->allocations.push_back(std::move(allocation));
    return true;
}

static void * required_tensor(spec_runtime * runtime, const std::string & name) {
    const auto it = runtime->tensors.find(name);
    return it == runtime->tensors.end() ? nullptr : it->second;
}

static bool load_mtp_artifact(spec_runtime * runtime) {
    gguf_init_params params {};
    params.no_alloc = true;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_file(runtime->artifact_path.c_str(), params), gguf_free);
    if (!gguf) {
        runtime->error = "cannot read accelerator GGUF: " + runtime->artifact_path;
        return false;
    }

    const int64_t arch_id = key_id(gguf.get(), "general.architecture", GGUF_TYPE_STRING);
    const int64_t version_id = key_id(gguf.get(), "beellama.spec_accel.version", GGUF_TYPE_UINT32);
    const int64_t kind_id = key_id(gguf.get(), "beellama.spec_accel.kind", GGUF_TYPE_STRING);
    const int64_t fingerprint_id = key_id(gguf.get(), "beellama.spec_accel.target_fingerprint", GGUF_TYPE_STRING);
    const int64_t ids_id = key_id(gguf.get(), "beellama.spec_accel.draft_vocab_ids", GGUF_TYPE_ARRAY);
    if (arch_id < 0 || std::strcmp(gguf_get_val_str(gguf.get(), arch_id), "beellama-spec-accel") != 0 ||
            version_id < 0 || gguf_get_val_u32(gguf.get(), version_id) != SPEC_ARTIFACT_VERSION ||
            kind_id < 0 || std::strcmp(gguf_get_val_str(gguf.get(), kind_id), "mtp") != 0 ||
            fingerprint_id < 0 || ids_id < 0 ||
            gguf_get_arr_type(gguf.get(), ids_id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(gguf.get(), ids_id) != N_HEAD_ROWS) {
        runtime->error = "accelerator GGUF metadata is missing, malformed, or from an unsupported version";
        return false;
    }

    std::string sampled;
    if (!sample_target_fingerprint(runtime->target_path.c_str(), sampled, runtime->error)) {
        return false;
    }
    const std::string artifact_fingerprint = gguf_get_val_str(gguf.get(), fingerprint_id);
    if (sampled != artifact_fingerprint ||
            (!runtime->model_fingerprint.empty() && runtime->model_fingerprint != artifact_fingerprint)) {
        runtime->error = "accelerator GGUF was prepared from a different target model (expected " +
                sampled + ", artifact " + artifact_fingerprint + ")";
        return false;
    }
    runtime->model_fingerprint = artifact_fingerprint;

    const auto schema = mtp_schema(SPEC_MTP_BLOCK);
    if (gguf_get_n_tensors(gguf.get()) != int64_t(schema.size())) {
        runtime->error = "MTP accelerator GGUF must contain exactly 17 tensors";
        return false;
    }
    for (const auto & expected : schema) {
        const int64_t id = gguf_find_tensor(gguf.get(), expected.name.c_str());
        if (id < 0 || gguf_get_tensor_type(gguf.get(), id) != expected.type) {
            runtime->error = "missing or incompatible accelerator tensor: " + expected.name;
            return false;
        }
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        if (ne[0] != expected.ne0 || ne[1] != expected.ne1) {
            runtime->error = "accelerator tensor shape mismatch: " + expected.name;
            return false;
        }
    }

    std::vector<int32_t> remap(N_HEAD_ROWS);
    std::memcpy(remap.data(), gguf_get_arr_data(gguf.get(), ids_id), remap.size() * sizeof(int32_t));
    std::vector<uint8_t> seen(SPEC_VOCAB, 0);
    for (int32_t token : remap) {
        if (token < 0 || token >= SPEC_VOCAB || seen[token] != 0) {
            runtime->error = "accelerator draft-vocabulary remap is invalid or non-injective";
            return false;
        }
        seen[token] = 1;
    }

    FILE * file = std::fopen(runtime->artifact_path.c_str(), "rb");
    if (file == nullptr) {
        runtime->error = "cannot open accelerator GGUF tensor data";
        return false;
    }
    for (const auto & expected : schema) {
        const int64_t id = gguf_find_tensor(gguf.get(), expected.name.c_str());
        if (!upload_tensor(runtime, file, gguf.get(), id, expected.name == "token_embd.weight")) {
            std::fclose(file);
            return false;
        }
    }
    std::fclose(file);

    runtime->remap = static_cast<int32_t *>(device_alloc(
            runtime, remap.size() * sizeof(int32_t), runtime->upload_stream));
    if (runtime->remap == nullptr ||
            !hip_ok(runtime, hipMemcpyAsync(runtime->remap, remap.data(), remap.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice, runtime->upload_stream), "upload draft-vocabulary remap") ||
            !hip_ok(runtime, hipStreamSynchronize(runtime->upload_stream), "synchronize remap upload")) {
        return false;
    }

    const auto q4 = [&](const std::string & name) {
        return static_cast<const block_q4_0 *>(required_tensor(runtime, name));
    };
    const auto f32 = [&](const std::string & name) {
        return static_cast<const float *>(required_tensor(runtime, name));
    };
    auto & w = runtime->weights;
    w.embd = q4("token_embd.weight");
    w.ehp = q4(tensor_name(SPEC_MTP_BLOCK, "nextn.eh_proj.weight"));
    w.enorm = f32(tensor_name(SPEC_MTP_BLOCK, "nextn.enorm.weight"));
    w.hnorm = f32(tensor_name(SPEC_MTP_BLOCK, "nextn.hnorm.weight"));
    w.attn = f32(tensor_name(SPEC_MTP_BLOCK, "attn_norm.weight"));
    w.q = q4(tensor_name(SPEC_MTP_BLOCK, "attn_q.weight"));
    w.k = q4(tensor_name(SPEC_MTP_BLOCK, "attn_k.weight"));
    w.v = q4(tensor_name(SPEC_MTP_BLOCK, "attn_v.weight"));
    w.o = q4(tensor_name(SPEC_MTP_BLOCK, "attn_output.weight"));
    w.qnorm = f32(tensor_name(SPEC_MTP_BLOCK, "attn_q_norm.weight"));
    w.knorm = f32(tensor_name(SPEC_MTP_BLOCK, "attn_k_norm.weight"));
    w.post = f32(tensor_name(SPEC_MTP_BLOCK, "post_attention_norm.weight"));
    w.fg = q4(tensor_name(SPEC_MTP_BLOCK, "ffn_gate.weight"));
    w.fu = q4(tensor_name(SPEC_MTP_BLOCK, "ffn_up.weight"));
    w.fd = q4(tensor_name(SPEC_MTP_BLOCK, "ffn_down.weight"));
    w.shnorm = f32(tensor_name(SPEC_MTP_BLOCK, "nextn.shared_head_norm.weight"));
    w.head = q4(tensor_name(SPEC_MTP_BLOCK, "nextn.shared_head_head.weight"));
    return true;
}

template <typename T>
static bool alloc_array(spec_sequence * sequence, T *& pointer, size_t count, const char * name) {
    pointer = static_cast<T *>(device_alloc(sequence->runtime, count * sizeof(T), sequence->stream));
    if (pointer == nullptr) {
        sequence->runtime->error = std::string("failed to allocate ") + name;
        return false;
    }
    return true;
}

static void free_mtp_scratch(spec_sequence * sequence) {
    auto & c = sequence->scratch;
    void * pointers[] = {
        c.x_tok, c.x_h, c.e_n, c.h_n, c.cat, c.proj, c.attin, c.qfull, c.q, c.gate,
        c.k, c.v, c.attout, c.ffn_in, c.ffn_g, c.ffn_u, c.ffn_tmp, c.resid, c.hnorm2,
        c.token, c.pos4,
    };
    for (void * pointer : pointers) {
        device_free(sequence->runtime, pointer, sequence->stream);
    }
    c = {};

    auto & b = sequence->batch;
    void * batch_pointers[] = {
        b.x_tok, b.h, b.e_n, b.h_n, b.cat, b.proj, b.attin, b.k, b.v, b.tokens, b.pos4,
        b.cache_rows,
        b.gemm_input,
    };
    for (void * pointer : batch_pointers) {
        device_free(sequence->runtime, pointer, sequence->stream);
    }
    b = {};

    device_free(sequence->runtime, sequence->part_m, sequence->stream);
    device_free(sequence->runtime, sequence->part_l, sequence->stream);
    device_free(sequence->runtime, sequence->part_acc, sequence->stream);
    device_free(sequence->runtime, sequence->logits, sequence->stream);
    device_free(sequence->runtime, sequence->pmax, sequence->stream);
    device_free(sequence->runtime, sequence->pidx, sequence->stream);
    device_free(sequence->runtime, sequence->token_out, sequence->stream);
    sequence->part_m = nullptr;
    sequence->part_l = nullptr;
    sequence->part_acc = nullptr;
    sequence->logits = nullptr;
    sequence->pmax = nullptr;
    sequence->pidx = nullptr;
    sequence->token_out = nullptr;
}

static bool allocate_mtp_scratch(spec_sequence * sequence) {
    auto & c = sequence->scratch;
    bool ok =
        alloc_array(sequence, c.x_tok, N_EMBD, "MTP token row") &&
        alloc_array(sequence, c.x_h, N_EMBD, "MTP hidden row") &&
        alloc_array(sequence, c.e_n, N_EMBD, "MTP normalized token row") &&
        alloc_array(sequence, c.h_n, N_EMBD, "MTP normalized hidden row") &&
        alloc_array(sequence, c.cat, 2 * N_EMBD, "MTP concatenation") &&
        alloc_array(sequence, c.proj, N_EMBD, "MTP projection") &&
        alloc_array(sequence, c.attin, N_EMBD, "MTP attention input") &&
        alloc_array(sequence, c.qfull, 2 * N_HEAD * HEAD_DIM, "MTP Q/gate") &&
        alloc_array(sequence, c.q, N_HEAD * HEAD_DIM, "MTP Q") &&
        alloc_array(sequence, c.gate, N_HEAD * HEAD_DIM, "MTP gate") &&
        alloc_array(sequence, c.k, N_HEAD_KV * HEAD_DIM, "MTP K") &&
        alloc_array(sequence, c.v, N_HEAD_KV * HEAD_DIM, "MTP V") &&
        alloc_array(sequence, c.attout, N_HEAD * HEAD_DIM, "MTP attention output") &&
        alloc_array(sequence, c.ffn_in, N_EMBD, "MTP FFN input") &&
        alloc_array(sequence, c.ffn_g, N_FF, "MTP FFN gate") &&
        alloc_array(sequence, c.ffn_u, N_FF, "MTP FFN up") &&
        alloc_array(sequence, c.ffn_tmp, N_EMBD, "MTP FFN output") &&
        alloc_array(sequence, c.resid, N_EMBD, "MTP residual") &&
        alloc_array(sequence, c.hnorm2, N_EMBD, "MTP next hidden") &&
        alloc_array(sequence, c.token, 1, "MTP token id") &&
        alloc_array(sequence, c.pos4, 4, "MTP M-RoPE position");

    sequence->n_parts = (N_HEAD_ROWS + 256 * 8 - 1) / (256 * 8);
    ok = ok &&
        alloc_array(sequence, sequence->part_m, N_HEAD * ATTN_MAX_PARTS, "MTP attention maxima") &&
        alloc_array(sequence, sequence->part_l, N_HEAD * ATTN_MAX_PARTS, "MTP attention sums") &&
        alloc_array(sequence, sequence->part_acc,
                size_t(N_HEAD) * ATTN_MAX_PARTS * HEAD_DIM, "MTP attention partials") &&
        alloc_array(sequence, sequence->logits, N_HEAD_ROWS, "MTP sliced logits") &&
        alloc_array(sequence, sequence->pmax, sequence->n_parts, "MTP argmax maxima") &&
        alloc_array(sequence, sequence->pidx, sequence->n_parts, "MTP argmax indices") &&
        alloc_array(sequence, sequence->token_out, 8, "MTP output tokens");
    if (!ok) {
        free_mtp_scratch(sequence);
    }
    return ok;
}

static bool ensure_batch_capacity(spec_sequence * sequence, uint32_t capacity) {
    auto & b = sequence->batch;
    if (capacity <= b.capacity) {
        return true;
    }

    void * pointers[] = {
        b.x_tok, b.h, b.e_n, b.h_n, b.cat, b.proj, b.attin, b.k, b.v, b.tokens, b.pos4,
        b.cache_rows,
        b.gemm_input,
    };
    for (void * pointer : pointers) {
        device_free(sequence->runtime, pointer, sequence->stream);
    }
    b = {};

    // Exact request size keeps this adaptive to the caller's actual ubatch;
    // no 128/256/512 route is baked into the accelerator.
    b.capacity = capacity;
    const size_t n = capacity;
    const bool ok =
        alloc_array(sequence, b.x_tok, n * N_EMBD, "batched MTP token rows") &&
        alloc_array(sequence, b.h, n * N_EMBD, "batched MTP hidden rows") &&
        alloc_array(sequence, b.e_n, n * N_EMBD, "batched MTP normalized tokens") &&
        alloc_array(sequence, b.h_n, n * N_EMBD, "batched MTP normalized hidden") &&
        alloc_array(sequence, b.cat, n * 2 * N_EMBD, "batched MTP concatenation") &&
        alloc_array(sequence, b.proj, n * N_EMBD, "batched MTP projection") &&
        alloc_array(sequence, b.attin, n * N_EMBD, "batched MTP attention input") &&
        alloc_array(sequence, b.k, n * N_HEAD_KV * HEAD_DIM, "batched MTP K") &&
        alloc_array(sequence, b.v, n * N_HEAD_KV * HEAD_DIM, "batched MTP V") &&
        alloc_array(sequence, b.tokens, n, "batched MTP tokens") &&
        alloc_array(sequence, b.pos4, n * 4, "batched MTP M-RoPE positions") &&
        alloc_array(sequence, b.cache_rows, n, "batched MTP cache rows") &&
        alloc_array(sequence, b.gemm_input, n * 2 * N_EMBD, "batched MTP GEMM input");
    if (!ok) {
        void * failed[] = {
            b.x_tok, b.h, b.e_n, b.h_n, b.cat, b.proj, b.attin, b.k, b.v, b.tokens, b.pos4,
            b.cache_rows,
            b.gemm_input,
        };
        for (void * pointer : failed) {
            device_free(sequence->runtime, pointer, sequence->stream);
        }
        b = {};
        return false;
    }
    return true;
}

static bool ensure_mtp_dense_weights(spec_runtime * runtime) {
    std::lock_guard<std::mutex> lock(runtime->dense_mutex);
    auto & dense = runtime->dense_weights;
    if (dense.ready) {
        return true;
    }
    if (dense.unavailable) {
        return false;
    }

    constexpr size_t ehp_elements = size_t(N_EMBD) * 2 * N_EMBD;
    constexpr size_t kv_elements = size_t(N_HEAD_KV) * HEAD_DIM * N_EMBD;
    __half * ehp = static_cast<__half *>(device_alloc(
            runtime, ehp_elements * sizeof(__half), runtime->upload_stream));
    __half * k = static_cast<__half *>(device_alloc(
            runtime, kv_elements * sizeof(__half), runtime->upload_stream));
    __half * v = static_cast<__half *>(device_alloc(
            runtime, kv_elements * sizeof(__half), runtime->upload_stream));
    const auto cleanup = [&]() {
        device_free(runtime, ehp, runtime->upload_stream);
        device_free(runtime, k, runtime->upload_stream);
        device_free(runtime, v, runtime->upload_stream);
        hipStreamSynchronize(runtime->upload_stream);
        dense.unavailable = true;
        // Dense catch-up is an optional optimization.  Allocation pressure
        // falls back to the quantized kernels without disabling speculation.
        runtime->error.clear();
        global_error.clear();
    };
    if (ehp == nullptr || k == nullptr || v == nullptr) {
        cleanup();
        return false;
    }

    constexpr int threads = 256;
    q4_0_matrix_to_f16<<<(ehp_elements + threads - 1) / threads, threads, 0, runtime->upload_stream>>>(
            runtime->weights.ehp, ehp, ehp_elements, 2 * N_EMBD);
    q4_0_matrix_to_f16<<<(kv_elements + threads - 1) / threads, threads, 0, runtime->upload_stream>>>(
            runtime->weights.k, k, kv_elements, N_EMBD);
    q4_0_matrix_to_f16<<<(kv_elements + threads - 1) / threads, threads, 0, runtime->upload_stream>>>(
            runtime->weights.v, v, kv_elements, N_EMBD);
    if (!hip_ok(runtime, hipGetLastError(), "materialize MTP FP16 projections") ||
            !hip_ok(runtime, hipStreamSynchronize(runtime->upload_stream),
                "synchronize MTP projection materialization")) {
        cleanup();
        return false;
    }

    dense.ehp = ehp;
    dense.k = k;
    dense.v = v;
    dense.ready = true;
    runtime->allocations.push_back({"dense.mtp.eh_proj.f16", ehp, nullptr, ehp_elements * sizeof(__half)});
    runtime->allocations.push_back({"dense.mtp.k.f16", k, nullptr, kv_elements * sizeof(__half)});
    runtime->allocations.push_back({"dense.mtp.v.f16", v, nullptr, kv_elements * sizeof(__half)});
    return true;
}

static bool mtp_dense_gemm(
        spec_sequence * sequence,
        const __half * weights,
        const float * input,
        float * output,
        int n_in,
        int n_out,
        int rows) {
    auto * runtime = sequence->runtime;
    auto & batch = sequence->batch;
    if (sequence->blas == nullptr || batch.gemm_input == nullptr || weights == nullptr) {
        runtime->error = "MTP dense catch-up route is not initialized";
        return false;
    }
    const size_t elements = size_t(rows) * n_in;
    constexpr int threads = 256;
    f32_matrix_to_f16<<<(elements + threads - 1) / threads, threads, 0, sequence->stream>>>(
            input, batch.gemm_input, elements);
    if (!hip_ok(runtime, hipGetLastError(), "convert MTP catch-up input to FP16")) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Matrices are stored row-major.  Interpreting the same bytes as
    // column-major yields C[n_out, rows] = transpose(W[n_in, n_out]) *
    // X[n_in, rows], which is byte-identical to the desired row-major output.
    return hipblas_ok(runtime, hipblasGemmEx(
            sequence->blas,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            n_out, rows, n_in,
            &alpha,
            weights, HIP_R_16F, n_in,
            batch.gemm_input, HIP_R_16F, n_in,
            &beta,
            output, HIP_R_32F, n_out,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
            "run batched MTP FP16 GEMM");
}

static bool ensure_cache_capacity(spec_sequence * sequence, size_t required) {
    if (required <= sequence->cache_capacity) {
        return true;
    }
    auto * runtime = sequence->runtime;
    size_t capacity = std::max(required, size_t(runtime->descriptor.n_ctx));
    capacity = std::min(capacity, size_t(ATTN_MAX_PARTS) * ATTN_SPLIT);
    if (capacity < required) {
        runtime->error = "requested MTP cache position exceeds the accelerator limit";
        return false;
    }

    const size_t bytes = cache_bytes(runtime, capacity);
    void * next_k = device_alloc(runtime, bytes, sequence->stream);
    void * next_v = device_alloc(runtime, bytes, sequence->stream);
    if (next_k == nullptr || next_v == nullptr) {
        device_free(runtime, next_k, sequence->stream);
        device_free(runtime, next_v, sequence->stream);
        runtime->error = "insufficient VRAM for the per-sequence MTP cache";
        return false;
    }
    if (sequence->cache_capacity > 0) {
        const size_t old_bytes = cache_bytes(runtime, sequence->cache_capacity);
        if (!hip_ok(runtime, hipMemcpyAsync(next_k, sequence->k_cache, old_bytes,
                        hipMemcpyDeviceToDevice, sequence->stream), "grow MTP K cache") ||
                !hip_ok(runtime, hipMemcpyAsync(next_v, sequence->v_cache, old_bytes,
                        hipMemcpyDeviceToDevice, sequence->stream), "grow MTP V cache") ||
                !hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize MTP cache growth")) {
            device_free(runtime, next_k, sequence->stream);
            device_free(runtime, next_v, sequence->stream);
            return false;
        }
    }
    device_free(runtime, sequence->k_cache, sequence->stream);
    device_free(runtime, sequence->v_cache, sequence->stream);
    sequence->k_cache = next_k;
    sequence->v_cache = next_v;
    sequence->cache_capacity = capacity;
    return true;
}

static void free_dflash_storage(spec_sequence * sequence);

static void free_sequence_storage(spec_sequence * sequence) {
    if (sequence == nullptr) {
        return;
    }
    clear_prepared_state(sequence);
    free_mtp_scratch(sequence);
    free_dflash_storage(sequence);
    device_free(sequence->runtime, sequence->k_cache, sequence->stream);
    device_free(sequence->runtime, sequence->v_cache, sequence->stream);
    sequence->k_cache = nullptr;
    sequence->v_cache = nullptr;
    sequence->cache_capacity = 0;
    if (sequence->blas != nullptr) {
        hipblasDestroy(sequence->blas);
        sequence->blas = nullptr;
    }
    if (sequence->stream != nullptr) {
        hipStreamSynchronize(sequence->stream);
        hipStreamDestroy(sequence->stream);
        sequence->stream = nullptr;
    }
}

static std::array<int32_t, 4> mtp_input_position(
        const ggml_spec_accel_mtp_catchup * input,
        uint32_t row) {
    std::array<int32_t, 4> result {};
    for (uint32_t plane = 0; plane < result.size(); ++plane) {
        result[plane] = input->positions[size_t(plane) * input->count + row];
    }
    return result;
}

static size_t mtp_catchup_record_begin(
        spec_sequence * sequence,
        const ggml_spec_accel_mtp_catchup * input) {
    if (sequence->mtp_positions.empty()) {
        return 0;
    }

    const auto first = mtp_input_position(input, 0);
    const auto exact = std::find(
            sequence->mtp_positions.begin(), sequence->mtp_positions.end(), first);
    if (exact != sequence->mtp_positions.end()) {
        // Target verification re-evaluates the anchor/draft suffix at the same
        // four-plane position.  Overwrite from that exact record rather than
        // appending another speculative copy.
        return size_t(exact - sequence->mtp_positions.begin());
    }

    // A split image ubatch legitimately appends another unique height/width
    // coordinate with the same temporal position.  Any backwards tuple that
    // is not already resident would instead create an unaddressable gap.
    if (first[0] >= sequence->mtp_positions.back()[0]) {
        return sequence->mtp_positions.size();
    }

    sequence->runtime->error =
            "MTP catch-up cannot place a non-resident backwards M-RoPE record";
    return std::numeric_limits<size_t>::max();
}

static bool launch_mtp_catchup(
        spec_sequence * sequence,
        const ggml_spec_accel_mtp_catchup * input) {
    auto * runtime = sequence->runtime;
    const uint32_t count = input->count;
    if (count == 0) {
        return true;
    }
    if (input->hidden_rows == nullptr || input->positions == nullptr ||
            (input->tokens == nullptr) == (input->token_embeddings == nullptr) ||
            input->hidden_width != N_EMBD || input->position_planes != 4 ||
            (input->token_embeddings != nullptr && input->token_embedding_width != N_EMBD)) {
        runtime->error = "invalid MTP catch-up input";
        return false;
    }

    const size_t record_begin = mtp_catchup_record_begin(sequence, input);
    if (record_begin == std::numeric_limits<size_t>::max() ||
            record_begin + count > runtime->descriptor.n_ctx) {
        if (runtime->error.empty()) {
            runtime->error = "MTP catch-up records exceed the configured context";
        }
        return false;
    }

    int32_t model_end = 0;
    std::vector<std::array<int32_t, 4>> incoming_positions;
    try {
        incoming_positions.reserve(count);
    } catch (const std::bad_alloc &) {
        runtime->error = "cannot allocate MTP catch-up position metadata";
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto position = mtp_input_position(input, i);
        if (input->tokens != nullptr && (input->tokens[i] < 0 || input->tokens[i] >= SPEC_VOCAB)) {
            runtime->error = "MTP catch-up token is outside the Qwen3.8 vocabulary";
            return false;
        }
        for (uint32_t plane = 0; plane < 4; ++plane) {
            const int32_t pos = position[plane];
            if (pos < 0 || uint32_t(pos) >= runtime->descriptor.n_ctx) {
                runtime->error = "MTP catch-up position is outside the configured context";
                return false;
            }
            model_end = std::max(model_end, pos + 1);
        }
        incoming_positions.push_back(position);
    }

    // Prepare the complete host-side transaction before touching K/V.  This
    // keeps allocation failures from leaving device cache rows committed with
    // no matching record-to-position metadata.
    std::vector<std::array<int32_t, 4>> final_positions;
    try {
        final_positions.reserve(record_begin + count);
        final_positions.insert(final_positions.end(), sequence->mtp_positions.begin(),
                sequence->mtp_positions.begin() + record_begin);
        final_positions.insert(final_positions.end(), incoming_positions.begin(), incoming_positions.end());
    } catch (const std::bad_alloc &) {
        runtime->error = "cannot allocate MTP catch-up record metadata";
        return false;
    }
    if (!ensure_cache_capacity(sequence, record_begin + count)) {
        return false;
    }

    // Size from the actual incoming ubatch.  Under memory pressure, divide it
    // by a free-VRAM-derived workspace bound instead of matching any named
    // -ub preset (128/256/512/etc.).
    constexpr size_t scratch_bytes_per_row =
            size_t(8 * N_EMBD + 2 * N_HEAD_KV * HEAD_DIM) * sizeof(float) +
            size_t(2 * N_EMBD) * sizeof(__half) + 5 * sizeof(int32_t);
    size_t free_vram = 0;
    size_t total_vram = 0;
    uint32_t rows_by_memory = count;
    if (hipMemGetInfo(&free_vram, &total_vram) == hipSuccess) {
        rows_by_memory = uint32_t(std::max<size_t>(
                SPEC_CATCHUP_TILE, std::min<size_t>(count, (free_vram / 8) / scratch_bytes_per_row)));
    } else {
        hipGetLastError();
    }
    const uint32_t tile_limit = std::min(count, rows_by_memory);
    if (!ensure_batch_capacity(sequence, tile_limit)) {
        return false;
    }

    std::vector<int32_t> cache_rows;
    try {
        cache_rows.resize(tile_limit);
    } catch (const std::bad_alloc &) {
        runtime->error = "cannot allocate MTP catch-up cache-row metadata";
        return false;
    }

    const auto & w = runtime->weights;
    auto & b = sequence->batch;
    constexpr uint32_t dense_min_rows = 32;
    const bool dense_available = tile_limit >= dense_min_rows && sequence->blas != nullptr &&
            ensure_mtp_dense_weights(runtime);
    for (uint32_t row0 = 0; row0 < count; row0 += tile_limit) {
        const uint32_t rows = std::min(tile_limit, count - row0);
        const bool use_dense = dense_available && rows >= dense_min_rows;
        const size_t row_bytes = size_t(rows) * N_EMBD * sizeof(float);
        for (uint32_t row = 0; row < rows; ++row) {
            cache_rows[row] = int32_t(record_begin + row0 + row);
        }
        if (!hip_ok(runtime, hipMemcpyAsync(b.h, input->hidden_rows + size_t(row0) * N_EMBD,
                        row_bytes, hipMemcpyHostToDevice, sequence->stream), "upload MTP hidden rows") ||
                !hip_ok(runtime, hipMemcpyAsync(b.pos4, input->positions + row0,
                        rows * sizeof(int32_t), hipMemcpyHostToDevice, sequence->stream),
                        "upload MTP temporal positions") ||
                !hip_ok(runtime, hipMemcpyAsync(b.cache_rows, cache_rows.data(),
                        rows * sizeof(int32_t), hipMemcpyHostToDevice, sequence->stream),
                        "upload MTP cache record indices")) {
            return false;
        }
        for (uint32_t plane = 1; plane < 4; ++plane) {
            if (!hip_ok(runtime, hipMemcpyAsync(b.pos4 + size_t(plane) * rows,
                            input->positions + size_t(plane) * count + row0,
                            rows * sizeof(int32_t), hipMemcpyHostToDevice, sequence->stream),
                            "upload MTP M-RoPE positions")) {
                return false;
            }
        }

        if (input->tokens != nullptr) {
            if (!hip_ok(runtime, hipMemcpyAsync(b.tokens, input->tokens + row0,
                            rows * sizeof(int32_t), hipMemcpyHostToDevice, sequence->stream),
                            "upload MTP tokens")) {
                return false;
            }
            deq_rows_q4_0_b<<<dim3((N_EMBD + 255) / 256, rows), 256, 0, sequence->stream>>>(
                    w.embd, b.tokens, b.x_tok, N_EMBD);
        } else if (!hip_ok(runtime, hipMemcpyAsync(b.x_tok,
                        input->token_embeddings + size_t(row0) * N_EMBD,
                        row_bytes, hipMemcpyHostToDevice, sequence->stream),
                        "upload MTP token embeddings")) {
            return false;
        }

        rmsnorm_b<<<rows, 256, 0, sequence->stream>>>(b.x_tok, w.enorm, b.e_n, N_EMBD);
        rmsnorm_b<<<rows, 256, 0, sequence->stream>>>(b.h, w.hnorm, b.h_n, N_EMBD);
        concat2_b<<<dim3((2 * N_EMBD + 255) / 256, rows), 256, 0, sequence->stream>>>(
                b.e_n, b.h_n, b.cat, N_EMBD);
        if (use_dense) {
            if (!mtp_dense_gemm(sequence, runtime->dense_weights.ehp,
                        b.cat, b.proj, 2 * N_EMBD, N_EMBD, rows)) {
                return false;
            }
        } else {
            gemv_q4_0_b<<<dim3(N_EMBD, rows), 32, 0, sequence->stream>>>(
                    w.ehp, b.cat, b.proj, 2 * N_EMBD, N_EMBD);
        }
        rmsnorm_b<<<rows, 256, 0, sequence->stream>>>(b.proj, w.attn, b.attin, N_EMBD);
        if (use_dense) {
            if (!mtp_dense_gemm(sequence, runtime->dense_weights.k,
                        b.attin, b.k, N_EMBD, N_HEAD_KV * HEAD_DIM, rows) ||
                    !mtp_dense_gemm(sequence, runtime->dense_weights.v,
                        b.attin, b.v, N_EMBD, N_HEAD_KV * HEAD_DIM, rows)) {
                return false;
            }
        } else {
            gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, rows), 32, 0, sequence->stream>>>(
                    w.k, b.attin, b.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
            gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, rows), 32, 0, sequence->stream>>>(
                    w.v, b.attin, b.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
        }
        rmsnorm_heads_b<<<dim3(N_HEAD_KV, rows), 256, 0, sequence->stream>>>(
                b.k, w.knorm, b.k, N_HEAD_KV);
        rope_mrope4_b<<<dim3(N_HEAD_KV, rows), N_ROT / 2, 0, sequence->stream>>>(
                b.k, N_HEAD_KV, b.pos4, rows);
        if (runtime->descriptor.cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0) {
            kv_store_q8_b<<<dim3(N_HEAD_KV * HEAD_DIM / 32, rows), 32, 0, sequence->stream>>>(
                    b.k, b.v,
                    static_cast<spec_block_q8_0 *>(sequence->k_cache),
                    static_cast<spec_block_q8_0 *>(sequence->v_cache), b.cache_rows, 0);
        } else {
            kv_store_b<<<dim3((N_HEAD_KV * HEAD_DIM + 255) / 256, rows), 256, 0, sequence->stream>>>(
                    b.k, b.v,
                    static_cast<__half *>(sequence->k_cache),
                    static_cast<__half *>(sequence->v_cache), b.cache_rows);
        }

        if (!hip_ok(runtime, hipGetLastError(), "launch MTP catch-up kernels") ||
                !hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize MTP catch-up")) {
            return false;
        }
    }

    sequence->mtp_positions = std::move(final_positions);
    sequence->has_multimodal_positions = std::any_of(
            sequence->mtp_positions.begin(), sequence->mtp_positions.end(),
            [](const std::array<int32_t, 4> & position) {
                return position[1] != position[0] || position[2] != position[0] ||
                        position[3] != position[0];
            });
    model_end = 0;
    for (const auto & position : sequence->mtp_positions) {
        for (int32_t value : position) {
            model_end = std::max(model_end, value + 1);
        }
    }
    sequence->coverage_begin = sequence->mtp_positions.empty() ? model_end :
            sequence->mtp_positions.front()[0];
    sequence->coverage_end = model_end;
    sequence->logical_pos = model_end;
    sequence->generation++;
    return true;
}

static void launch_mtp_phase1(
        spec_sequence * sequence,
        const float * hidden,
        int32_t cache_row) {
    const auto & w = sequence->runtime->weights;
    auto & c = sequence->scratch;
    hipStream_t stream = sequence->stream;

    rmsnorm<<<1, 256, 0, stream>>>(c.x_tok, w.enorm, c.e_n, N_EMBD);
    rmsnorm<<<1, 256, 0, stream>>>(hidden, w.hnorm, c.h_n, N_EMBD);
    concat2<<<(2 * N_EMBD + 255) / 256, 256, 0, stream>>>(c.e_n, c.h_n, c.cat, N_EMBD);
    gemv_q4_0<<<N_EMBD, 32, 0, stream>>>(w.ehp, c.cat, c.proj, 2 * N_EMBD, N_EMBD);
    rmsnorm<<<1, 256, 0, stream>>>(c.proj, w.attn, c.attin, N_EMBD);
    gemv_q4_0<<<2 * N_HEAD * HEAD_DIM, 32, 0, stream>>>(
            w.q, c.attin, c.qfull, N_EMBD, 2 * N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, stream>>>(
            w.k, c.attin, c.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, stream>>>(
            w.v, c.attin, c.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
    split_q_gate<<<N_HEAD, HEAD_DIM, 0, stream>>>(c.qfull, c.q, c.gate);
    rmsnorm_heads<<<N_HEAD, 256, 0, stream>>>(c.q, w.qnorm, c.q, N_HEAD);
    rmsnorm_heads<<<N_HEAD_KV, 256, 0, stream>>>(c.k, w.knorm, c.k, N_HEAD_KV);
    rope_mrope4<<<N_HEAD, N_ROT / 2, 0, stream>>>(c.q, N_HEAD, c.pos4);
    rope_mrope4<<<N_HEAD_KV, N_ROT / 2, 0, stream>>>(c.k, N_HEAD_KV, c.pos4);
    if (sequence->runtime->descriptor.cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0) {
        kv_store_q8_b<<<dim3(N_HEAD_KV * HEAD_DIM / 32, 1), 32, 0, stream>>>(
                c.k, c.v,
                static_cast<spec_block_q8_0 *>(sequence->k_cache),
                static_cast<spec_block_q8_0 *>(sequence->v_cache), nullptr, cache_row);
    } else {
        kv_store<<<(N_HEAD_KV * HEAD_DIM + 255) / 256, 256, 0, stream>>>(
                c.k, c.v,
                static_cast<__half *>(sequence->k_cache),
                static_cast<__half *>(sequence->v_cache), cache_row);
    }
}

static void launch_mtp_phase2(spec_sequence * sequence, int32_t n_records, int32_t * output) {
    const auto & w = sequence->runtime->weights;
    auto & c = sequence->scratch;
    hipStream_t stream = sequence->stream;
    const int attention_parts = (n_records + ATTN_SPLIT - 1) / ATTN_SPLIT;

    if (sequence->runtime->descriptor.cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0) {
        attn_1q_split_q8<<<dim3(N_HEAD, attention_parts), 256, 0, stream>>>(
                c.q,
                static_cast<const spec_block_q8_0 *>(sequence->k_cache),
                static_cast<const spec_block_q8_0 *>(sequence->v_cache), n_records,
                sequence->part_m, sequence->part_l, sequence->part_acc);
    } else {
        attn_1q_split<<<dim3(N_HEAD, attention_parts), 256, 0, stream>>>(
                c.q,
                static_cast<const __half *>(sequence->k_cache),
                static_cast<const __half *>(sequence->v_cache), n_records,
                sequence->part_m, sequence->part_l, sequence->part_acc);
    }
    attn_merge<<<N_HEAD, HEAD_DIM, 0, stream>>>(
            sequence->part_m, sequence->part_l, sequence->part_acc, c.attout, attention_parts);
    sigmoid_gate<<<(N_HEAD * HEAD_DIM + 255) / 256, 256, 0, stream>>>(
            c.attout, c.gate, N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_EMBD, 32, 0, stream>>>(w.o, c.attout, c.resid, N_HEAD * HEAD_DIM, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, stream>>>(c.resid, c.proj, N_EMBD);
    rmsnorm<<<1, 256, 0, stream>>>(c.resid, w.post, c.ffn_in, N_EMBD);
    gemv_q4_0<<<N_FF, 32, 0, stream>>>(w.fg, c.ffn_in, c.ffn_g, N_EMBD, N_FF);
    gemv_q4_0<<<N_FF, 32, 0, stream>>>(w.fu, c.ffn_in, c.ffn_u, N_EMBD, N_FF);
    silu_mul<<<(N_FF + 255) / 256, 256, 0, stream>>>(c.ffn_g, c.ffn_u, N_FF);
    gemv_q4_0<<<N_EMBD, 32, 0, stream>>>(w.fd, c.ffn_g, c.ffn_tmp, N_FF, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, stream>>>(c.resid, c.ffn_tmp, N_EMBD);
    rmsnorm<<<1, 256, 0, stream>>>(c.resid, w.shnorm, c.hnorm2, N_EMBD);
    gemv_q4_0<<<N_HEAD_ROWS, 32, 0, stream>>>(w.head, c.hnorm2, sequence->logits, N_EMBD, N_HEAD_ROWS);
    argmax_partial<<<sequence->n_parts, 256, 0, stream>>>(
            sequence->logits, N_HEAD_ROWS, sequence->pmax, sequence->pidx);
    argmax_final<<<1, 1, 0, stream>>>(
            sequence->pmax, sequence->pidx, sequence->n_parts, N_HEAD_ROWS,
            sequence->runtime->remap, SPEC_VOCAB, output);
}

static bool launch_mtp_draft(spec_sequence * sequence, ggml_spec_accel_mtp_draft * io) {
    auto * runtime = sequence->runtime;
    if (io->hidden == nullptr || io->hidden_width != N_EMBD || io->max_draft < 1 ||
            io->max_draft > 8 || io->output_ids == nullptr || io->output_capacity < io->max_draft ||
            io->last_token < 0 || io->last_token >= SPEC_VOCAB || io->past_records < 0 ||
            io->next_position < 0) {
        runtime->error = "invalid MTP draft request";
        return false;
    }
    if (uint64_t(io->past_records) + io->max_draft > runtime->descriptor.n_ctx ||
            uint64_t(io->next_position) + io->max_draft > runtime->descriptor.n_ctx ||
            !ensure_cache_capacity(sequence, size_t(io->past_records) + io->max_draft)) {
        return false;
    }
    if (sequence->coverage_begin > 0 || sequence->coverage_end < io->next_position ||
            sequence->mtp_positions.size() < size_t(io->past_records)) {
        runtime->error = "MTP accelerator cache does not cover the requested prefix";
        return false;
    }

    try {
        sequence->mtp_positions.reserve(size_t(io->past_records) + io->max_draft);
    } catch (const std::bad_alloc &) {
        runtime->error = "cannot allocate MTP draft record metadata";
        return false;
    }

    auto & c = sequence->scratch;
    if (!hip_ok(runtime, hipMemcpyAsync(c.x_h, io->hidden, N_EMBD * sizeof(float),
                    hipMemcpyHostToDevice, sequence->stream), "upload MTP draft hidden row") ||
            !hip_ok(runtime, hipMemcpyAsync(c.token, &io->last_token, sizeof(int32_t),
                    hipMemcpyHostToDevice, sequence->stream), "upload MTP draft token")) {
        return false;
    }

    // The caller's accepted record count and next model position are
    // authoritative after verification/rollback.  This makes a rejected
    // speculative suffix a logical truncation without conflating image rows
    // with the compressed M-RoPE temporal position.
    sequence->mtp_positions.resize(size_t(io->past_records));
    sequence->coverage_end = io->next_position;
    sequence->logical_pos = io->next_position;

    std::array<int32_t, 4> position {};
    for (uint32_t step = 0; step < io->max_draft; ++step) {
        const int32_t pos = io->next_position + int32_t(step);
        const int32_t cache_row = io->past_records + int32_t(step);
        position.fill(pos);
        if (!hip_ok(runtime, hipMemcpyAsync(c.pos4, position.data(), sizeof(position),
                        hipMemcpyHostToDevice, sequence->stream), "upload MTP draft M-RoPE position")) {
            return false;
        }
        const int32_t * token = step == 0 ? c.token : sequence->token_out + step - 1;
        const float * hidden = step == 0 ? c.x_h : c.hnorm2;
        deq_row_q4_0<<<(N_EMBD + 255) / 256, 256, 0, sequence->stream>>>(
                runtime->weights.embd, token, c.x_tok, N_EMBD, SPEC_VOCAB);
        launch_mtp_phase1(sequence, hidden, cache_row);
        launch_mtp_phase2(sequence, cache_row + 1, sequence->token_out + step);
        sequence->mtp_positions.push_back(position);
    }
    if (!hip_ok(runtime, hipGetLastError(), "launch MTP draft kernels") ||
            !hip_ok(runtime, hipMemcpyAsync(io->output_ids, sequence->token_out,
                    io->max_draft * sizeof(int32_t), hipMemcpyDeviceToHost, sequence->stream),
                    "download MTP draft tokens") ||
            !hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize MTP draft")) {
        return false;
    }
    for (uint32_t i = 0; i < io->max_draft; ++i) {
        if (io->output_ids[i] < 0 || io->output_ids[i] >= SPEC_VOCAB) {
            runtime->error = "MTP accelerator produced an invalid token";
            return false;
        }
    }
    io->output_count = io->max_draft;
    sequence->coverage_end = io->next_position + int32_t(io->max_draft);
    sequence->logical_pos = sequence->coverage_end;
    sequence->generation++;
    return true;
}

__global__ void shift_mrope_k_f16(__half * cache, size_t n_positions, int32_t delta) {
    const size_t row = blockIdx.x;
    const int head = blockIdx.y;
    const int pair = threadIdx.x;
    if (row >= n_positions || head >= N_HEAD_KV || pair >= N_ROT / 2) {
        return;
    }
    __half * k = cache + (row * N_HEAD_KV + head) * HEAD_DIM;
    const float theta = delta * powf(FREQ_BASE, -2.0f * pair / N_ROT);
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float x0 = __half2float(k[pair]);
    const float x1 = __half2float(k[pair + N_ROT / 2]);
    k[pair] = __float2half(x0 * c - x1 * s);
    k[pair + N_ROT / 2] = __float2half(x0 * s + x1 * c);
}

__global__ void shift_mrope_k_q8(
        spec_block_q8_0 * cache,
        size_t n_positions,
        int32_t delta) {
    const size_t row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= n_positions || head >= N_HEAD_KV || lane >= HEAD_DIM) {
        return;
    }
    __shared__ float values[HEAD_DIM];
    const size_t base = (row * N_HEAD_KV + head) * HEAD_DIM;
    values[lane] = spec_q8_get(cache, base + lane);
    __syncthreads();

    if (lane < N_ROT / 2) {
        const float theta = delta * powf(FREQ_BASE, -2.0f * lane / N_ROT);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = values[lane];
        const float x1 = values[lane + N_ROT / 2];
        values[lane] = x0 * c - x1 * s;
        values[lane + N_ROT / 2] = x0 * s + x1 * c;
    }
    __syncthreads();

    float maximum = fabsf(values[lane]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 32));
    }
    maximum = __shfl(maximum, 0, 32);
    const float scale = maximum / 127.0f;
    spec_block_q8_0 * block = cache + base / 32 + lane / 32;
    if ((lane & 31) == 0) {
        block->d = __float2half(scale);
    }
    block->qs[lane & 31] = maximum == 0.0f ? 0 : spec_q8_quantize(values[lane], 1.0f / scale);
}

#include "dflash-runtime.inc"

static int32_t api_query_capabilities_impl(
        const ggml_spec_accel_descriptor * descriptor,
        ggml_spec_accel_capabilities * capabilities) {
    if (!valid_struct(descriptor, descriptor ? descriptor->struct_size : 0,
                sizeof(ggml_spec_accel_descriptor)) ||
            !valid_struct(capabilities, capabilities ? capabilities->struct_size : 0,
                sizeof(ggml_spec_accel_capabilities))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    const bool mtp = descriptor->kind == GGML_SPEC_ACCEL_KIND_MTP;
    const bool dflash = descriptor->kind == GGML_SPEC_ACCEL_KIND_DFLASH;
    if ((!mtp && !dflash) || descriptor->n_embd != N_EMBD ||
            descriptor->n_vocab != SPEC_VOCAB || descriptor->n_head != N_HEAD ||
            descriptor->n_head_kv != N_HEAD_KV || descriptor->head_dim != HEAD_DIM ||
            descriptor->n_rot != N_ROT ||
            (mtp && descriptor->cache_type != GGML_SPEC_ACCEL_CACHE_F16 &&
                    descriptor->cache_type != GGML_SPEC_ACCEL_CACHE_Q8_0) ||
            (dflash && descriptor->cache_type != GGML_SPEC_ACCEL_CACHE_F16) ||
            descriptor->n_ctx == 0 || descriptor->n_ctx > 131072 ||
            descriptor->n_seq_max == 0) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }

    int device = descriptor->device_ordinal;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        hipGetLastError();
        return GGML_SPEC_ACCEL_STATUS_UNAVAILABLE;
    }
    if (device < 0) {
        device = 0;
    }
    if (device >= count) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    hipDeviceProp_t properties {};
    if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
        hipGetLastError();
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    const std::string arch = properties.gcnArchName;
    if (arch.rfind("gfx1100", 0) != 0) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }

    capabilities->abi_version = GGML_SPEC_ACCEL_ABI_VERSION;
    capabilities->flags =
            GGML_SPEC_ACCEL_CAP_TOKEN_EMBEDDINGS |
            GGML_SPEC_ACCEL_CAP_MROPE4 |
            GGML_SPEC_ACCEL_CAP_REMOVE_SUFFIX |
            GGML_SPEC_ACCEL_CAP_COPY_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_SHIFT_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_CHECKPOINT |
            GGML_SPEC_ACCEL_CAP_DURABLE_STATE |
            GGML_SPEC_ACCEL_CAP_MULTIMODAL_ROWS;
    if (mtp) {
        capabilities->flags |= GGML_SPEC_ACCEL_CAP_MTP |
                GGML_SPEC_ACCEL_CAP_CACHE_F16 | GGML_SPEC_ACCEL_CAP_CACHE_Q8_0;
    } else {
        capabilities->flags |= GGML_SPEC_ACCEL_CAP_DFLASH | GGML_SPEC_ACCEL_CAP_CACHE_F16;
    }
    hipMemPool_t pool = nullptr;
    if (hipDeviceGetDefaultMemPool(&pool, device) == hipSuccess && pool != nullptr) {
        capabilities->flags |= GGML_SPEC_ACCEL_CAP_ASYNC_ALLOC;
    } else {
        hipGetLastError();
    }
    capabilities->max_sequences = descriptor->n_seq_max;
    capabilities->max_context = std::min<uint32_t>(descriptor->n_ctx, 131072);
    capabilities->preferred_catchup_rows = mtp ? 512 : 128;
    capabilities->feature_width = 0;
    capabilities->block_size = 0;
    capabilities->cache_window = 0;
    capabilities->target_layer_count = 0;
    for (int32_t & layer : capabilities->target_layers) {
        layer = 0;
    }
    if (dflash) {
        std::string error;
        if (!query_dflash_artifact_descriptor(descriptor->artifact_path, capabilities, error)) {
            global_error = error;
            return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
        }
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_query_capabilities(
        const ggml_spec_accel_descriptor * descriptor,
        ggml_spec_accel_capabilities * capabilities) noexcept {
    try {
        return api_query_capabilities_impl(descriptor, capabilities);
    } catch (const std::bad_alloc &) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception & error) {
        try {
            global_error = error.what();
        } catch (...) {
        }
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    } catch (...) {
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
}

static int32_t api_runtime_create_impl(
        const ggml_spec_accel_descriptor * descriptor,
        ggml_spec_accel_runtime_t * output) {
    if (output == nullptr) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    *output = nullptr;
    ggml_spec_accel_capabilities capabilities {};
    capabilities.struct_size = sizeof(capabilities);
    const int32_t status = api_query_capabilities(descriptor, &capabilities);
    if (status != GGML_SPEC_ACCEL_STATUS_OK) {
        return status;
    }
    if (descriptor->artifact_path == nullptr || descriptor->artifact_path[0] == '\0' ||
            descriptor->target_model_path == nullptr || descriptor->target_model_path[0] == '\0') {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }

    const auto cleanup_runtime = [](spec_runtime * runtime) {
        if (runtime == nullptr) {
            return;
        }
        free_tensor_allocations(runtime);
        if (runtime->upload_stream != nullptr) {
            hipStreamSynchronize(runtime->upload_stream);
            hipStreamDestroy(runtime->upload_stream);
            runtime->upload_stream = nullptr;
        }
        delete runtime;
    };
    std::unique_ptr<spec_runtime, decltype(cleanup_runtime)> runtime(
            new (std::nothrow) spec_runtime, cleanup_runtime);
    if (!runtime) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    runtime->artifact_path = descriptor->artifact_path;
    runtime->target_path = descriptor->target_model_path;
    runtime->model_fingerprint = descriptor->model_fingerprint ? descriptor->model_fingerprint : "";
    runtime->device_name = descriptor->device_name ? descriptor->device_name : "";
    runtime->descriptor = *descriptor;
    runtime->descriptor.artifact_path = runtime->artifact_path.c_str();
    runtime->descriptor.target_model_path = runtime->target_path.c_str();
    runtime->descriptor.model_fingerprint = runtime->model_fingerprint.c_str();
    runtime->descriptor.device_name = runtime->device_name.c_str();
    runtime->device = descriptor->device_ordinal < 0 ? 0 : descriptor->device_ordinal;

    if (!hip_ok(runtime.get(), hipSetDevice(runtime->device), "select HIP speculative device") ||
            !hip_ok(runtime.get(), hipStreamCreateWithFlags(&runtime->upload_stream, hipStreamNonBlocking),
                    "create HIP upload stream")) {
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    hipMemPool_t pool = nullptr;
    runtime->async_alloc = hipDeviceGetDefaultMemPool(&pool, runtime->device) == hipSuccess && pool != nullptr;
    if (runtime->async_alloc) {
        uint64_t threshold = std::numeric_limits<uint64_t>::max();
        if (hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold) != hipSuccess) {
            hipGetLastError();
        }
    } else {
        hipGetLastError();
    }

    const bool loaded = descriptor->kind == GGML_SPEC_ACCEL_KIND_MTP ?
            load_mtp_artifact(runtime.get()) : load_dflash_artifact(runtime.get());
    if (!loaded) {
        global_error = runtime->error;
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    runtime->descriptor.model_fingerprint = runtime->model_fingerprint.c_str();
    *output = runtime.release();
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_runtime_create(
        const ggml_spec_accel_descriptor * descriptor,
        ggml_spec_accel_runtime_t * output) noexcept {
    try {
        return api_runtime_create_impl(descriptor, output);
    } catch (const std::bad_alloc &) {
        if (output != nullptr) {
            *output = nullptr;
        }
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception & error) {
        if (output != nullptr) {
            *output = nullptr;
        }
        try {
            global_error = error.what();
        } catch (...) {
        }
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    } catch (...) {
        if (output != nullptr) {
            *output = nullptr;
        }
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
}

static void api_sequence_free(ggml_spec_accel_seq_t handle);

static void api_runtime_free(ggml_spec_accel_runtime_t handle) {
    auto * runtime = static_cast<spec_runtime *>(handle);
    if (runtime == nullptr) {
        return;
    }
    for (;;) {
        spec_sequence * sequence = nullptr;
        {
            std::lock_guard<std::mutex> lock(runtime->mutex);
            if (runtime->sequences.empty()) {
                break;
            }
            sequence = runtime->sequences.begin()->second;
        }
        api_sequence_free(sequence);
    }
    free_tensor_allocations(runtime);
    if (runtime->upload_stream != nullptr) {
        hipStreamSynchronize(runtime->upload_stream);
        hipStreamDestroy(runtime->upload_stream);
    }
    delete runtime;
}

static int32_t api_sequence_create(
        ggml_spec_accel_runtime_t runtime_handle,
        int32_t sequence_id,
        ggml_spec_accel_seq_t * output) {
    auto * runtime = static_cast<spec_runtime *>(runtime_handle);
    if (runtime == nullptr || output == nullptr || sequence_id < 0) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    *output = nullptr;
    // Reject duplicate/over-capacity slots before allocating streams and
    // potentially hundreds of MiB of per-sequence scratch. Recheck again at
    // insertion time to preserve correctness if callers create concurrently.
    {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->sequences.size() >= runtime->descriptor.n_seq_max ||
                runtime->sequences.find(sequence_id) != runtime->sequences.end()) {
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_INVALID,
                    "accelerator sequence id is duplicate or exceeds the configured slot count");
        }
    }
    std::unique_ptr<spec_sequence> sequence(new (std::nothrow) spec_sequence);
    if (!sequence) {
        return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY, "cannot allocate accelerator sequence state");
    }
    sequence->runtime = runtime;
    sequence->id = sequence_id;
    if (runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP) {
        try {
            sequence->pending_hidden.resize(N_EMBD, 0.0f);
        } catch (const std::bad_alloc &) {
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                    "cannot allocate accelerator sequence metadata");
        }
    }
    if (!hip_ok(runtime, hipSetDevice(runtime->device), "select sequence HIP device") ||
            !hip_ok(runtime, hipStreamCreateWithFlags(&sequence->stream, hipStreamNonBlocking),
                    "create accelerator sequence stream")) {
        free_sequence_storage(sequence.get());
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    if (hipblasCreate(&sequence->blas) != HIPBLAS_STATUS_SUCCESS ||
            hipblasSetStream(sequence->blas, sequence->stream) != HIPBLAS_STATUS_SUCCESS) {
        if (sequence->blas != nullptr) {
            hipblasDestroy(sequence->blas);
            sequence->blas = nullptr;
        }
        // hipBLAS is an adaptive large-row optimization for MTP catch-up and
        // DFlash feature injection.  Their quantized kernels remain complete
        // fallbacks on older runtimes or under resource pressure.
    }
    if (!(runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP ?
                allocate_mtp_scratch(sequence.get()) : allocate_dflash_storage(sequence.get()))) {
        free_sequence_storage(sequence.get());
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->sequences.size() >= runtime->descriptor.n_seq_max ||
                runtime->sequences.find(sequence_id) != runtime->sequences.end()) {
            free_sequence_storage(sequence.get());
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_INVALID,
                    "accelerator sequence id is duplicate or exceeds the configured slot count");
        }
        try {
            runtime->sequences.emplace(sequence_id, sequence.get());
        } catch (const std::bad_alloc &) {
            free_sequence_storage(sequence.get());
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                    "cannot register accelerator sequence state");
        }
    }
    *output = sequence.release();
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static void api_sequence_free(ggml_spec_accel_seq_t handle) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr) {
        return;
    }
    auto * runtime = sequence->runtime;
    if (runtime != nullptr) {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        const auto it = runtime->sequences.find(sequence->id);
        if (it != runtime->sequences.end() && it->second == sequence) {
            runtime->sequences.erase(it);
        }
    }
    free_sequence_storage(sequence);
    delete sequence;
}

static int32_t api_sequence_reset(ggml_spec_accel_seq_t handle) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    sequence->logical_pos = 0;
    sequence->coverage_begin = 0;
    sequence->coverage_end = 0;
    sequence->mtp_positions.clear();
    sequence->generation++;
    std::fill(sequence->pending_hidden.begin(), sequence->pending_hidden.end(), 0.0f);
    sequence->has_multimodal_positions = false;
    sequence->healthy = true;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_sequence_invalidate(ggml_spec_accel_seq_t handle, int32_t logical_pos) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || logical_pos < 0 ||
            uint32_t(logical_pos) > sequence->runtime->descriptor.n_ctx) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    sequence->logical_pos = logical_pos;
    sequence->coverage_begin = logical_pos;
    sequence->coverage_end = logical_pos;
    sequence->mtp_positions.clear();
    sequence->generation++;
    std::fill(sequence->pending_hidden.begin(), sequence->pending_hidden.end(), 0.0f);
    sequence->has_multimodal_positions = false;
    sequence->healthy = true;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_mtp_catchup(
        ggml_spec_accel_seq_t handle,
        const ggml_spec_accel_mtp_catchup * input) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(input, input ? input->struct_size : 0, sizeof(*input))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (sequence->runtime->descriptor.kind != GGML_SPEC_ACCEL_KIND_MTP) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }
    if (input->count > 0 &&
            (input->hidden_rows == nullptr || input->positions == nullptr ||
             (input->tokens == nullptr) == (input->token_embeddings == nullptr) ||
             input->hidden_width != N_EMBD || input->position_planes != 4 ||
             (input->token_embeddings != nullptr && input->token_embedding_width != N_EMBD))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    for (uint32_t row = 0; row < input->count; ++row) {
        if (input->tokens != nullptr &&
                (input->tokens[row] < 0 || input->tokens[row] >= SPEC_VOCAB)) {
            return GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        for (uint32_t plane = 0; plane < 4; ++plane) {
            const int32_t position = input->positions[size_t(plane) * input->count + row];
            if (position < 0 || uint32_t(position) >= sequence->runtime->descriptor.n_ctx) {
                return GGML_SPEC_ACCEL_STATUS_INVALID;
            }
        }
    }
    if (!sequence->healthy) {
        return GGML_SPEC_ACCEL_STATUS_UNAVAILABLE;
    }
    if (!launch_mtp_catchup(sequence, input)) {
        sequence->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_mtp_draft(
        ggml_spec_accel_seq_t handle,
        ggml_spec_accel_mtp_draft * input_output) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(input_output,
                input_output ? input_output->struct_size : 0, sizeof(*input_output))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (sequence->runtime->descriptor.kind != GGML_SPEC_ACCEL_KIND_MTP) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }
    if (input_output->hidden == nullptr || input_output->hidden_width != N_EMBD ||
            input_output->max_draft < 1 || input_output->max_draft > 8 ||
            input_output->output_ids == nullptr ||
            input_output->output_capacity < input_output->max_draft ||
            input_output->last_token < 0 || input_output->last_token >= SPEC_VOCAB ||
            input_output->past_records < 0 || input_output->next_position < 0 ||
            uint64_t(input_output->past_records) + input_output->max_draft >
                sequence->runtime->descriptor.n_ctx ||
            uint64_t(input_output->next_position) + input_output->max_draft >
                sequence->runtime->descriptor.n_ctx) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (!sequence->healthy) {
        return GGML_SPEC_ACCEL_STATUS_UNAVAILABLE;
    }
    if (sequence->coverage_begin > 0 ||
            sequence->coverage_end < input_output->next_position ||
            sequence->mtp_positions.size() < size_t(input_output->past_records)) {
        return GGML_SPEC_ACCEL_STATUS_NO_COVERAGE;
    }
    if (!launch_mtp_draft(sequence, input_output)) {
        sequence->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_dflash_features(
        ggml_spec_accel_seq_t handle,
        const ggml_spec_accel_dflash_features * input) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(input,
                input ? input->struct_size : 0, sizeof(*input))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (sequence->runtime->descriptor.kind != GGML_SPEC_ACCEL_KIND_DFLASH) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }
    if (input->count > 0 && (input->features == nullptr || input->positions == nullptr ||
            input->feature_width != 25600 || input->position_planes != 4)) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    int32_t previous = -1;
    for (uint32_t row = 0; row < input->count; ++row) {
        const int32_t position = input->positions[row];
        if (position < 0 || uint32_t(position) >= sequence->runtime->descriptor.n_ctx ||
                (previous >= 0 && position != previous + 1)) {
            return GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        previous = position;
        for (uint32_t plane = 1; plane < 4; ++plane) {
            const int32_t plane_position = input->positions[size_t(plane) * input->count + row];
            if (plane_position < 0 ||
                    uint32_t(plane_position) >= sequence->runtime->descriptor.n_ctx) {
                return GGML_SPEC_ACCEL_STATUS_INVALID;
            }
        }
    }
    if (!sequence->healthy) {
        return GGML_SPEC_ACCEL_STATUS_UNAVAILABLE;
    }
    if (!launch_dflash_features(sequence, input)) {
        sequence->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_dflash_draft(
        ggml_spec_accel_seq_t handle,
        ggml_spec_accel_dflash_draft * input_output) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(input_output,
                input_output ? input_output->struct_size : 0, sizeof(*input_output))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (sequence->runtime->descriptor.kind != GGML_SPEC_ACCEL_KIND_DFLASH) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }
    if (input_output->last_token < 0 || input_output->last_token >= SPEC_VOCAB ||
            input_output->past_tokens < 1 || input_output->max_draft == 0 ||
            input_output->max_draft > DF_BLOCK - 1 || input_output->output_ids == nullptr ||
            input_output->output_capacity < input_output->max_draft ||
            (input_output->output_confidences != nullptr &&
                input_output->confidence_capacity < input_output->max_draft) ||
            uint64_t(input_output->past_tokens) + DF_BLOCK >
                sequence->runtime->descriptor.n_ctx) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (!sequence->healthy) {
        return GGML_SPEC_ACCEL_STATUS_UNAVAILABLE;
    }
    const int32_t required_begin = std::max(0, input_output->past_tokens - (DF_WINDOW - 1));
    if (sequence->coverage_begin > required_begin ||
            sequence->coverage_end < input_output->past_tokens) {
        return GGML_SPEC_ACCEL_STATUS_NO_COVERAGE;
    }
    if (!launch_dflash_draft(sequence, input_output)) {
        sequence->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_sequence_plan(
        ggml_spec_accel_seq_t handle,
        ggml_spec_accel_sequence_plan * plan) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(plan, plan ? plan->struct_size : 0, sizeof(*plan))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    plan->logical_pos = sequence->logical_pos;
    plan->coverage_begin = sequence->coverage_begin;
    plan->coverage_end = sequence->coverage_end;
    plan->generation = sequence->generation;
    plan->can_remove_suffix = sequence->healthy ? 1u : 0u;
    plan->can_shift = sequence->healthy && !sequence->has_multimodal_positions ? 1u : 0u;
    plan->record_count = sequence->runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP ?
            uint32_t(sequence->mtp_positions.size()) :
            uint32_t(std::max(0, sequence->coverage_end - sequence->coverage_begin));
    plan->reserved = 0;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_sequence_remove_suffix(ggml_spec_accel_seq_t handle, int32_t pos) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || pos < 0) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (sequence->runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP) {
        const auto boundary = std::lower_bound(
                sequence->mtp_positions.begin(), sequence->mtp_positions.end(), pos,
                [](const std::array<int32_t, 4> & position, int32_t value) {
                    return position[0] < value;
                });
        sequence->mtp_positions.resize(
                size_t(boundary - sequence->mtp_positions.begin()));
        sequence->has_multimodal_positions = std::any_of(
                sequence->mtp_positions.begin(), sequence->mtp_positions.end(),
                [](const std::array<int32_t, 4> & position) {
                    return position[1] != position[0] || position[2] != position[0] ||
                            position[3] != position[0];
                });
    }
    if (pos < sequence->coverage_begin) {
        // No resident row survives, but preserve the logical boundary.  A
        // zero-based range here would falsely claim coverage from the start
        // of a long context and could make a ring cache read stale slots.
        sequence->coverage_begin = pos;
        sequence->coverage_end = pos;
    } else {
        sequence->coverage_end = std::min(sequence->coverage_end, pos);
    }
    sequence->logical_pos = std::min(sequence->logical_pos, pos);
    sequence->generation++;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_sequence_copy(
        ggml_spec_accel_seq_t source_handle,
        ggml_spec_accel_seq_t destination_handle,
        int32_t begin,
        int32_t end) {
    auto * source = static_cast<spec_sequence *>(source_handle);
    auto * destination = static_cast<spec_sequence *>(destination_handle);
    if (source == nullptr || destination == nullptr || source->runtime != destination->runtime ||
            begin < 0 || end < begin || source->coverage_begin > begin || source->coverage_end < end) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    if (begin == end) {
        return GGML_SPEC_ACCEL_STATUS_OK;
    }
    auto * runtime = source->runtime;
    if (runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH) {
        const int32_t count = end - begin;
        if (count > DF_WINDOW) {
            return GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        if (source == destination) {
            return GGML_SPEC_ACCEL_STATUS_OK;
        }
        if (!hip_ok(runtime, hipStreamSynchronize(source->stream),
                    "synchronize source DFlash sequence copy")) {
            return GGML_SPEC_ACCEL_STATUS_RUNTIME;
        }
        const size_t bytes = size_t(count) * dflash_cache_row_bytes();
        void * temp = device_alloc(runtime, bytes, destination->stream);
        if (temp == nullptr) {
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                    "cannot allocate DFlash fork workspace");
        }
        bool ok = true;
        for (int layer = 0; layer < DF_LAYERS && ok; ++layer) {
            ok = dflash_ring_to_contiguous(runtime, source->dflash.k_cache[layer], begin, count,
                            temp, hipMemcpyDeviceToDevice, destination->stream, "stage DFlash K fork") &&
                    dflash_contiguous_to_ring(runtime, temp, begin, count,
                            destination->dflash.k_cache[layer], hipMemcpyDeviceToDevice,
                            destination->stream, "commit DFlash K fork") &&
                    dflash_ring_to_contiguous(runtime, source->dflash.v_cache[layer], begin, count,
                            temp, hipMemcpyDeviceToDevice, destination->stream, "stage DFlash V fork") &&
                    dflash_contiguous_to_ring(runtime, temp, begin, count,
                            destination->dflash.v_cache[layer], hipMemcpyDeviceToDevice,
                            destination->stream, "commit DFlash V fork");
        }
        if (ok) {
            ok = hip_ok(runtime, hipStreamSynchronize(destination->stream),
                    "synchronize DFlash sequence copy");
        }
        device_free(runtime, temp, destination->stream);
        if (!ok) {
            destination->healthy = false;
            return GGML_SPEC_ACCEL_STATUS_RUNTIME;
        }
        destination->coverage_begin = begin;
        destination->coverage_end = end;
        destination->logical_pos = source->logical_pos >= begin && source->logical_pos <= end ?
                source->logical_pos : end;
        destination->has_multimodal_positions = source->has_multimodal_positions;
        destination->healthy = source->healthy;
        destination->generation++;
        return GGML_SPEC_ACCEL_STATUS_OK;
    }
    const auto row_begin_it = std::lower_bound(
            source->mtp_positions.begin(), source->mtp_positions.end(), begin,
            [](const std::array<int32_t, 4> & position, int32_t value) {
                return position[0] < value;
            });
    const auto row_end_it = std::lower_bound(
            source->mtp_positions.begin(), source->mtp_positions.end(), end,
            [](const std::array<int32_t, 4> & position, int32_t value) {
                return position[0] < value;
            });
    const size_t row_begin = size_t(row_begin_it - source->mtp_positions.begin());
    const size_t row_end = size_t(row_end_it - source->mtp_positions.begin());
    const size_t records = row_end - row_begin;
    if (source == destination) {
        return GGML_SPEC_ACCEL_STATUS_OK;
    }

    std::vector<std::array<int32_t, 4>> destination_positions;
    try {
        destination_positions.assign(row_begin_it, row_end_it);
    } catch (const std::bad_alloc &) {
        return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                "cannot allocate MTP fork record metadata");
    }
    if (!ensure_cache_capacity(destination, records)) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    const size_t bytes = cache_bytes(runtime, records);
    if (bytes > 0 &&
            (!hip_ok(runtime, hipMemcpyAsync(
                        destination->k_cache,
                        cache_at(source->k_cache, runtime, row_begin),
                        bytes, hipMemcpyDeviceToDevice, destination->stream), "copy MTP K sequence") ||
             !hip_ok(runtime, hipMemcpyAsync(
                        destination->v_cache,
                        cache_at(source->v_cache, runtime, row_begin),
                        bytes, hipMemcpyDeviceToDevice, destination->stream), "copy MTP V sequence") ||
             !hip_ok(runtime, hipStreamSynchronize(destination->stream), "synchronize MTP sequence copy"))) {
        destination->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    destination->mtp_positions = std::move(destination_positions);
    destination->coverage_begin = begin;
    destination->coverage_end = end;
    destination->logical_pos = source->logical_pos >= begin && source->logical_pos <= end ?
            source->logical_pos : end;
    destination->has_multimodal_positions = std::any_of(
            destination->mtp_positions.begin(), destination->mtp_positions.end(),
            [](const std::array<int32_t, 4> & position) {
                return position[1] != position[0] || position[2] != position[0] ||
                        position[3] != position[0];
            });
    destination->healthy = source->healthy;
    destination->generation++;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_sequence_shift(
        ggml_spec_accel_seq_t handle,
        int32_t begin,
        int32_t end,
        int32_t delta) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || begin < 0 || end < begin || delta == 0 ||
            sequence->coverage_begin > begin || sequence->coverage_end < end) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    // Delta-RoPE shifting is well-defined for ordinary scalar positions.  An
    // image batch can use different temporal/height/width planes, so keep the
    // same fail-closed policy as the server's multimodal context shifting.
    if (sequence->has_multimodal_positions) {
        return GGML_SPEC_ACCEL_STATUS_UNSUPPORTED;
    }
    const int64_t shifted_begin64 = int64_t(begin) + delta;
    const int64_t shifted_end64 = int64_t(end) + delta;
    if (shifted_begin64 < 0 || shifted_end64 < shifted_begin64 ||
            shifted_end64 > sequence->runtime->descriptor.n_ctx) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    const int32_t shifted_begin = int32_t(shifted_begin64);
    const int32_t shifted_end = int32_t(shifted_end64);
    auto * runtime = sequence->runtime;
    if (runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH) {
        const int32_t count = end - begin;
        if (count > DF_WINDOW) {
            return GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        if (count == 0) {
            return GGML_SPEC_ACCEL_STATUS_OK;
        }
        const size_t bytes = size_t(count) * dflash_cache_row_bytes();
        void * temp_k = device_alloc(runtime, bytes, sequence->stream);
        void * temp_v = device_alloc(runtime, bytes, sequence->stream);
        if (temp_k == nullptr || temp_v == nullptr) {
            device_free(runtime, temp_k, sequence->stream);
            device_free(runtime, temp_v, sequence->stream);
            return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                    "cannot allocate DFlash shift workspace");
        }
        bool ok = true;
        for (int layer = 0; layer < DF_LAYERS && ok; ++layer) {
            ok = dflash_ring_to_contiguous(runtime, sequence->dflash.k_cache[layer], begin, count,
                            temp_k, hipMemcpyDeviceToDevice, sequence->stream, "stage shifted DFlash K") &&
                    dflash_ring_to_contiguous(runtime, sequence->dflash.v_cache[layer], begin, count,
                            temp_v, hipMemcpyDeviceToDevice, sequence->stream, "stage shifted DFlash V") &&
                    dflash_contiguous_to_ring(runtime, temp_k, begin + delta, count,
                            sequence->dflash.k_cache[layer], hipMemcpyDeviceToDevice,
                            sequence->stream, "commit shifted DFlash K") &&
                    dflash_contiguous_to_ring(runtime, temp_v, begin + delta, count,
                            sequence->dflash.v_cache[layer], hipMemcpyDeviceToDevice,
                            sequence->stream, "commit shifted DFlash V");
            if (ok) {
                df_shift_k_ring<<<dim3(count, DF_KV), DF_HDIM / 2, 0, sequence->stream>>>(
                        sequence->dflash.k_cache[layer], begin + delta, count, delta,
                        runtime->dflash.rope_freq_base);
                ok = hip_ok(runtime, hipGetLastError(), "launch delta-RoPE DFlash shift");
            }
        }
        if (ok) {
            ok = hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize DFlash shift");
        }
        device_free(runtime, temp_k, sequence->stream);
        device_free(runtime, temp_v, sequence->stream);
        if (!ok) {
            sequence->healthy = false;
            return GGML_SPEC_ACCEL_STATUS_RUNTIME;
        }
        sequence->coverage_begin = begin + delta;
        sequence->coverage_end = shifted_end;
        if (sequence->logical_pos >= begin && sequence->logical_pos <= end) {
            sequence->logical_pos += delta;
        }
        sequence->generation++;
        return GGML_SPEC_ACCEL_STATUS_OK;
    }
    const auto source_begin = std::lower_bound(
            sequence->mtp_positions.begin(), sequence->mtp_positions.end(), begin,
            [](const std::array<int32_t, 4> & position, int32_t value) {
                return position[0] < value;
            });
    const auto source_end = std::lower_bound(
            sequence->mtp_positions.begin(), sequence->mtp_positions.end(), end,
            [](const std::array<int32_t, 4> & position, int32_t value) {
                return position[0] < value;
            });
    const auto prefix_end = std::lower_bound(
            sequence->mtp_positions.begin(), sequence->mtp_positions.end(), shifted_begin,
            [](const std::array<int32_t, 4> & position, int32_t value) {
                return position[0] < value;
            });
    const size_t source_row = size_t(source_begin - sequence->mtp_positions.begin());
    const size_t destination_row = size_t(prefix_end - sequence->mtp_positions.begin());
    const size_t positions = size_t(source_end - source_begin);

    std::vector<std::array<int32_t, 4>> shifted_positions;
    try {
        shifted_positions.reserve(destination_row + positions);
        shifted_positions.insert(shifted_positions.end(), sequence->mtp_positions.begin(), prefix_end);
        for (auto it = source_begin; it != source_end; ++it) {
            auto position = *it;
            for (int32_t & value : position) {
                value += delta;
            }
            shifted_positions.push_back(position);
        }
    } catch (const std::bad_alloc &) {
        return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY,
                "cannot allocate shifted MTP record metadata");
    }
    if (!ensure_cache_capacity(sequence, shifted_positions.size())) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    if (positions == 0) {
        sequence->mtp_positions = std::move(shifted_positions);
        sequence->coverage_begin = shifted_begin;
        sequence->coverage_end = shifted_begin;
        sequence->logical_pos = shifted_begin;
        sequence->generation++;
        return GGML_SPEC_ACCEL_STATUS_OK;
    }
    const size_t bytes = cache_bytes(runtime, positions);
    void * temp_k = device_alloc(runtime, bytes, sequence->stream);
    void * temp_v = device_alloc(runtime, bytes, sequence->stream);
    if (temp_k == nullptr || temp_v == nullptr) {
        device_free(runtime, temp_k, sequence->stream);
        device_free(runtime, temp_v, sequence->stream);
        return fail(runtime, GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY, "cannot allocate MTP shift workspace");
    }
    bool ok =
        hip_ok(runtime, hipMemcpyAsync(temp_k, cache_at(sequence->k_cache, runtime, source_row), bytes,
                hipMemcpyDeviceToDevice, sequence->stream), "stage shifted MTP K") &&
        hip_ok(runtime, hipMemcpyAsync(temp_v, cache_at(sequence->v_cache, runtime, source_row), bytes,
                hipMemcpyDeviceToDevice, sequence->stream), "stage shifted MTP V");
    if (ok) {
        if (runtime->descriptor.cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0) {
            shift_mrope_k_q8<<<dim3(positions, N_HEAD_KV), HEAD_DIM, 0, sequence->stream>>>(
                    static_cast<spec_block_q8_0 *>(temp_k), positions, delta);
        } else {
            shift_mrope_k_f16<<<dim3(positions, N_HEAD_KV), N_ROT / 2, 0, sequence->stream>>>(
                    static_cast<__half *>(temp_k), positions, delta);
        }
        ok = hip_ok(runtime, hipGetLastError(), "launch delta-RoPE MTP shift") &&
            hip_ok(runtime, hipMemcpyAsync(cache_at(sequence->k_cache, runtime, destination_row), temp_k, bytes,
                    hipMemcpyDeviceToDevice, sequence->stream), "commit shifted MTP K") &&
            hip_ok(runtime, hipMemcpyAsync(cache_at(sequence->v_cache, runtime, destination_row), temp_v, bytes,
                    hipMemcpyDeviceToDevice, sequence->stream), "commit shifted MTP V") &&
            hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize MTP shift");
    }
    device_free(runtime, temp_k, sequence->stream);
    device_free(runtime, temp_v, sequence->stream);
    if (!ok) {
        sequence->healthy = false;
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    sequence->mtp_positions = std::move(shifted_positions);
    sequence->coverage_begin = sequence->mtp_positions.empty() ? shifted_begin :
            sequence->mtp_positions.front()[0];
    sequence->coverage_end = 0;
    for (const auto & position : sequence->mtp_positions) {
        for (int32_t value : position) {
            sequence->coverage_end = std::max(sequence->coverage_end, value + 1);
        }
    }
    if (sequence->logical_pos >= begin && sequence->logical_pos <= end) {
        sequence->logical_pos += delta;
    }
    sequence->generation++;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_checkpoint_validate(
        ggml_spec_accel_seq_t handle,
        const ggml_spec_accel_checkpoint_desc * checkpoint,
        ggml_spec_accel_checkpoint_t * output) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || output == nullptr ||
            !valid_struct(checkpoint, checkpoint ? checkpoint->struct_size : 0, sizeof(*checkpoint))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    *output = nullptr;
    const bool mtp = sequence->runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP;
    const bool pending_valid = mtp ?
            (checkpoint->pending_hidden_count == 0 ||
                (checkpoint->pending_hidden != nullptr && checkpoint->pending_hidden_count == N_EMBD)) :
            checkpoint->pending_hidden_count == 0;
    if (checkpoint->logical_pos < 0 || checkpoint->coverage_begin < 0 ||
            checkpoint->coverage_end < checkpoint->coverage_begin ||
            checkpoint->logical_pos > checkpoint->coverage_end ||
            checkpoint->coverage_begin < sequence->coverage_begin ||
            checkpoint->coverage_end > sequence->coverage_end ||
            checkpoint->reserved != 0 ||
            (mtp && checkpoint->record_count > sequence->mtp_positions.size()) ||
            !pending_valid) {
        return GGML_SPEC_ACCEL_STATUS_NO_COVERAGE;
    }
    std::unique_ptr<checkpoint_transaction> transaction(new (std::nothrow) checkpoint_transaction);
    if (!transaction) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    transaction->sequence = sequence;
    transaction->descriptor = *checkpoint;
    if (checkpoint->pending_hidden_count != 0) {
        try {
            transaction->pending_hidden.assign(checkpoint->pending_hidden,
                    checkpoint->pending_hidden + checkpoint->pending_hidden_count);
        } catch (const std::bad_alloc &) {
            return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
        }
        transaction->descriptor.pending_hidden = transaction->pending_hidden.data();
    }
    *output = transaction.release();
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_checkpoint_commit(ggml_spec_accel_checkpoint_t handle) {
    auto * transaction = static_cast<checkpoint_transaction *>(handle);
    if (transaction == nullptr || transaction->sequence == nullptr) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    auto * sequence = transaction->sequence;
    const auto & checkpoint = transaction->descriptor;
    if (checkpoint.coverage_begin < sequence->coverage_begin ||
            checkpoint.coverage_end > sequence->coverage_end) {
        return GGML_SPEC_ACCEL_STATUS_NO_COVERAGE;
    }
    sequence->logical_pos = checkpoint.logical_pos;
    sequence->coverage_begin = checkpoint.coverage_begin;
    sequence->coverage_end = checkpoint.coverage_end;
    if (sequence->runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_MTP) {
        sequence->mtp_positions.resize(checkpoint.record_count);
        sequence->has_multimodal_positions = std::any_of(
                sequence->mtp_positions.begin(), sequence->mtp_positions.end(),
                [](const std::array<int32_t, 4> & position) {
                    return position[1] != position[0] || position[2] != position[0] ||
                            position[3] != position[0];
                });
    }
    sequence->generation = std::max(sequence->generation + 1, checkpoint.generation + 1);
    if (!transaction->pending_hidden.empty()) {
        sequence->pending_hidden = transaction->pending_hidden;
    }
    sequence->healthy = true;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static void api_checkpoint_free(ggml_spec_accel_checkpoint_t handle) {
    delete static_cast<checkpoint_transaction *>(handle);
}

static int32_t api_state_export(
        ggml_spec_accel_seq_t handle,
        ggml_spec_accel_blob * output) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    if (sequence == nullptr || !valid_struct(output, output ? output->struct_size : 0, sizeof(*output))) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    output->data = nullptr;
    output->size = 0;
    auto * runtime = sequence->runtime;
    const bool dflash = runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH;
    const int32_t resident = sequence->coverage_end - sequence->coverage_begin;
    if (resident < 0 || (dflash && resident > DF_WINDOW)) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    const size_t records = dflash ? size_t(resident) : sequence->mtp_positions.size();
    const size_t one_cache = dflash ? records * dflash_cache_row_bytes() :
            cache_bytes(runtime, records);
    const size_t pending_bytes = dflash ? 0 : sequence->pending_hidden.size() * sizeof(float);
    const size_t position_bytes = dflash ? 0 : records * sizeof(std::array<int32_t, 4>);
    const size_t cache_tensors = dflash ? size_t(DF_LAYERS) * 2 : 2;
    if (cache_tensors != 0 && one_cache >
                (std::numeric_limits<size_t>::max() - sizeof(state_header) - pending_bytes -
                    position_bytes) / cache_tensors) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    const size_t total = sizeof(state_header) + pending_bytes + position_bytes +
            cache_tensors * one_cache;
    auto * data = static_cast<uint8_t *>(std::malloc(total));
    if (data == nullptr) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    state_header header {
        SPEC_STATE_MAGIC,
        SPEC_STATE_VERSION,
        runtime->descriptor.kind,
        runtime->descriptor.cache_type,
        sequence->has_multimodal_positions ? SPEC_STATE_FLAG_MULTIMODAL_POSITIONS : 0u,
        0,
        sequence->logical_pos,
        sequence->coverage_begin,
        sequence->coverage_end,
        uint32_t(records),
        dflash ? 0u : uint32_t(sequence->pending_hidden.size()),
        position_bytes,
        sequence->generation,
        one_cache,
        0,
    };
    std::memcpy(data, &header, sizeof(header));
    if (pending_bytes > 0) {
        std::memcpy(data + sizeof(header), sequence->pending_hidden.data(), pending_bytes);
    }
    uint8_t * positions = data + sizeof(header) + pending_bytes;
    if (position_bytes > 0) {
        std::memcpy(positions, sequence->mtp_positions.data(), position_bytes);
    }
    uint8_t * cache = positions + position_bytes;
    bool copied = true;
    if (one_cache > 0 && dflash) {
        for (int layer = 0; layer < DF_LAYERS && copied; ++layer) {
            copied = dflash_ring_to_contiguous(runtime, sequence->dflash.k_cache[layer],
                            sequence->coverage_begin, resident, cache,
                            hipMemcpyDeviceToHost, sequence->stream, "export DFlash K state") &&
                    dflash_ring_to_contiguous(runtime, sequence->dflash.v_cache[layer],
                            sequence->coverage_begin, resident, cache + one_cache,
                            hipMemcpyDeviceToHost, sequence->stream, "export DFlash V state");
            cache += 2 * one_cache;
        }
    } else if (one_cache > 0) {
        copied = hip_ok(runtime, hipMemcpyAsync(cache, sequence->k_cache, one_cache,
                            hipMemcpyDeviceToHost, sequence->stream), "export MTP K state") &&
                hip_ok(runtime, hipMemcpyAsync(cache + one_cache, sequence->v_cache, one_cache,
                            hipMemcpyDeviceToHost, sequence->stream), "export MTP V state");
    }
    if (copied && one_cache > 0) {
        copied = hip_ok(runtime, hipStreamSynchronize(sequence->stream),
                dflash ? "synchronize DFlash state export" : "synchronize MTP state export");
    }
    if (!copied) {
        std::free(data);
        return GGML_SPEC_ACCEL_STATUS_RUNTIME;
    }
    uint64_t checksum = 1469598103934665603ULL;
    checksum = fnv1a(checksum, data, offsetof(state_header, checksum));
    checksum = fnv1a(checksum, data + sizeof(state_header), total - sizeof(state_header));
    std::memcpy(data + offsetof(state_header, checksum), &checksum, sizeof(checksum));
    output->data = data;
    output->size = total;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t validate_state_input(
        spec_sequence * sequence,
        const void * input,
        size_t size,
        state_header & header,
        bool reserve_capacity) {
    if (sequence == nullptr || input == nullptr || size < sizeof(state_header)) {
        return GGML_SPEC_ACCEL_STATUS_INVALID;
    }
    std::memcpy(&header, input, sizeof(header));
    auto * runtime = sequence->runtime;
    const bool dflash = runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH;
    const int64_t resident64 = int64_t(header.coverage_end) - header.coverage_begin;
    if (header.magic != SPEC_STATE_MAGIC || header.version != SPEC_STATE_VERSION ||
            header.kind != runtime->descriptor.kind ||
            header.cache_type != runtime->descriptor.cache_type ||
            (header.flags & ~SPEC_STATE_FLAG_MULTIMODAL_POSITIONS) != 0 ||
            (dflash ? header.pending_count != 0 : header.pending_count != N_EMBD) ||
            (dflash ? header.position_bytes != 0 :
                header.position_bytes != uint64_t(header.record_count) *
                    sizeof(std::array<int32_t, 4>)) ||
            header.logical_pos < 0 ||
            header.coverage_begin < 0 || header.coverage_end < header.coverage_begin ||
            header.logical_pos > header.coverage_end ||
            uint32_t(header.coverage_end) > runtime->descriptor.n_ctx ||
            header.record_count > runtime->descriptor.n_ctx ||
            (dflash && header.record_count != uint32_t(resident64)) ||
            (dflash && resident64 > DF_WINDOW)) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    const int32_t resident = int32_t(resident64);
    const size_t pending_bytes = size_t(header.pending_count) * sizeof(float);
    const size_t expected_cache = dflash ? size_t(resident) * dflash_cache_row_bytes() :
            cache_bytes(runtime, size_t(header.record_count));
    const size_t cache_tensors = dflash ? size_t(DF_LAYERS) * 2 : 2;
    const size_t size_max = std::numeric_limits<size_t>::max();
    if (header.position_bytes > size_max || header.cache_bytes > size_max ||
            pending_bytes > size_max - sizeof(header)) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    size_t expected_size = sizeof(header) + pending_bytes;
    const size_t position_bytes = size_t(header.position_bytes);
    if (position_bytes > size_max - expected_size) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    expected_size += position_bytes;
    const size_t serialized_cache = size_t(header.cache_bytes);
    if (serialized_cache != expected_cache ||
            (cache_tensors != 0 && serialized_cache > (size_max - expected_size) / cache_tensors)) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    expected_size += cache_tensors * serialized_cache;
    if (size != expected_size) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    if (!dflash && header.position_bytes > 0) {
        const auto * positions = reinterpret_cast<const std::array<int32_t, 4> *>(
                static_cast<const uint8_t *>(input) + sizeof(header) + pending_bytes);
        int32_t previous_temporal = -1;
        bool multimodal = false;
        for (uint32_t row = 0; row < header.record_count; ++row) {
            if (positions[row][0] < previous_temporal) {
                return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
            }
            previous_temporal = positions[row][0];
            for (uint32_t plane = 0; plane < 4; ++plane) {
                if (positions[row][plane] < 0 ||
                        uint32_t(positions[row][plane]) >= runtime->descriptor.n_ctx) {
                    return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
                }
                multimodal = multimodal || positions[row][plane] != positions[row][0];
            }
        }
        if (multimodal != ((header.flags & SPEC_STATE_FLAG_MULTIMODAL_POSITIONS) != 0)) {
            return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
        }
    }
    const auto * data = static_cast<const uint8_t *>(input);
    uint64_t checksum = 1469598103934665603ULL;
    checksum = fnv1a(checksum, data, offsetof(state_header, checksum));
    checksum = fnv1a(checksum, data + sizeof(state_header), size - sizeof(state_header));
    if (checksum != header.checksum) {
        return GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE;
    }
    if (reserve_capacity && !dflash && !ensure_cache_capacity(sequence, header.record_count)) {
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_state_validate(
        ggml_spec_accel_seq_t handle,
        const void * input,
        size_t size) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    state_header header {};
    const int32_t validation = validate_state_input(sequence, input, size, header, false);
    if (validation != GGML_SPEC_ACCEL_STATUS_OK) {
        return validation;
    }

    clear_prepared_state(sequence);
    auto * runtime = sequence->runtime;
    auto & prepared = sequence->prepared;
    const bool dflash = runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH;
    const int32_t resident = header.coverage_end - header.coverage_begin;
    const size_t pending_bytes = size_t(header.pending_count) * sizeof(float);
    const auto * data = static_cast<const uint8_t *>(input);
    const uint8_t * pending = data + sizeof(header);
    const uint8_t * positions = pending + pending_bytes;
    const uint8_t * cache = positions + header.position_bytes;

    try {
        if (!dflash) {
            const auto * pending_values = reinterpret_cast<const float *>(pending);
            prepared.pending_hidden.assign(
                    pending_values, pending_values + header.pending_count);
            const auto * position_values =
                    reinterpret_cast<const std::array<int32_t, 4> *>(positions);
            prepared.mtp_positions.assign(
                    position_values, position_values + header.record_count);
        }
    } catch (const std::bad_alloc &) {
        clear_prepared_state(sequence);
        return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
    }

    if (header.cache_bytes > 0) {
        if (dflash) {
            const size_t ring_bytes = size_t(DF_WINDOW) * dflash_cache_row_bytes();
            for (int layer = 0; layer < DF_LAYERS; ++layer) {
                prepared.k[layer] = device_alloc(runtime, ring_bytes, sequence->stream);
                prepared.v[layer] = device_alloc(runtime, ring_bytes, sequence->stream);
                if (prepared.k[layer] == nullptr || prepared.v[layer] == nullptr ||
                        !dflash_contiguous_to_ring(runtime, cache,
                            header.coverage_begin, resident,
                            static_cast<__half *>(prepared.k[layer]), hipMemcpyHostToDevice,
                            sequence->stream, "stage DFlash K state") ||
                        !dflash_contiguous_to_ring(runtime, cache + header.cache_bytes,
                            header.coverage_begin, resident,
                            static_cast<__half *>(prepared.v[layer]), hipMemcpyHostToDevice,
                            sequence->stream, "stage DFlash V state")) {
                    clear_prepared_state(sequence);
                    return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
                }
                cache += 2 * header.cache_bytes;
            }
        } else {
            prepared.k[0] = device_alloc(runtime, header.cache_bytes, sequence->stream);
            prepared.v[0] = device_alloc(runtime, header.cache_bytes, sequence->stream);
            if (prepared.k[0] == nullptr || prepared.v[0] == nullptr) {
                clear_prepared_state(sequence);
                return GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY;
            }
            prepared.mtp_capacity = size_t(header.record_count);
            if (!hip_ok(runtime, hipMemcpyAsync(prepared.k[0], cache, header.cache_bytes,
                            hipMemcpyHostToDevice, sequence->stream), "stage MTP K state") ||
                    !hip_ok(runtime, hipMemcpyAsync(prepared.v[0], cache + header.cache_bytes,
                            header.cache_bytes, hipMemcpyHostToDevice, sequence->stream),
                            "stage MTP V state")) {
                clear_prepared_state(sequence);
                return GGML_SPEC_ACCEL_STATUS_RUNTIME;
            }
        }
        if (!hip_ok(runtime, hipStreamSynchronize(sequence->stream),
                    dflash ? "synchronize staged DFlash state" : "synchronize staged MTP state")) {
            clear_prepared_state(sequence);
            return GGML_SPEC_ACCEL_STATUS_RUNTIME;
        }
    }

    prepared.input_size = size;
    prepared.input_checksum = header.checksum;
    prepared.source_generation = sequence->generation;
    prepared.saved_generation = header.generation;
    prepared.logical_pos = header.logical_pos;
    prepared.coverage_begin = header.coverage_begin;
    prepared.coverage_end = header.coverage_end;
    prepared.flags = header.flags;
    prepared.ready = true;
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static int32_t api_state_import(
        ggml_spec_accel_seq_t handle,
        const void * input,
        size_t size) {
    auto * sequence = static_cast<spec_sequence *>(handle);
    state_header header {};
    const int32_t validation = validate_state_input(sequence, input, size, header, false);
    if (validation != GGML_SPEC_ACCEL_STATUS_OK) {
        return validation;
    }
    if (!sequence->prepared.ready || sequence->prepared.input_size != size ||
            sequence->prepared.input_checksum != header.checksum ||
            sequence->prepared.source_generation != sequence->generation) {
        const int32_t staged = api_state_validate(handle, input, size);
        if (staged != GGML_SPEC_ACCEL_STATUS_OK) {
            return staged;
        }
    }
    auto * runtime = sequence->runtime;
    const bool dflash = runtime->descriptor.kind == GGML_SPEC_ACCEL_KIND_DFLASH;
    auto & prepared = sequence->prepared;
    // Validation already copied and synchronized every device byte.  Commit
    // only exchanges ownership, so target/draft restore cannot be followed by
    // a partially written speculative cache.
    if (header.cache_bytes > 0 && dflash) {
        if (sequence->dflash.graph_exec != nullptr) {
            hipGraphExecDestroy(sequence->dflash.graph_exec);
            sequence->dflash.graph_exec = nullptr;
        }
        for (int layer = 0; layer < DF_LAYERS; ++layer) {
            void * old_k = sequence->dflash.k_cache[layer];
            void * old_v = sequence->dflash.v_cache[layer];
            sequence->dflash.k_cache[layer] = static_cast<__half *>(prepared.k[layer]);
            sequence->dflash.v_cache[layer] = static_cast<__half *>(prepared.v[layer]);
            prepared.k[layer] = old_k;
            prepared.v[layer] = old_v;
        }
    } else if (header.cache_bytes > 0) {
        std::swap(sequence->k_cache, prepared.k[0]);
        std::swap(sequence->v_cache, prepared.v[0]);
        sequence->cache_capacity = prepared.mtp_capacity;
    }
    if (dflash) {
        sequence->pending_hidden.clear();
    } else {
        sequence->pending_hidden = std::move(prepared.pending_hidden);
        sequence->mtp_positions = std::move(prepared.mtp_positions);
    }
    sequence->logical_pos = prepared.logical_pos;
    sequence->coverage_begin = prepared.coverage_begin;
    sequence->coverage_end = prepared.coverage_end;
    sequence->generation = prepared.saved_generation + 1;
    sequence->has_multimodal_positions =
            (prepared.flags & SPEC_STATE_FLAG_MULTIMODAL_POSITIONS) != 0;
    sequence->healthy = true;
    clear_prepared_state(sequence);
    return GGML_SPEC_ACCEL_STATUS_OK;
}

static void api_state_cancel(ggml_spec_accel_seq_t handle) {
    clear_prepared_state(static_cast<spec_sequence *>(handle));
}

static void api_blob_free(ggml_spec_accel_blob * blob) {
    if (blob == nullptr) {
        return;
    }
    std::free(blob->data);
    blob->data = nullptr;
    blob->size = 0;
}

static const char * api_last_error(ggml_spec_accel_runtime_t handle) {
    auto * runtime = static_cast<spec_runtime *>(handle);
    return runtime == nullptr ? global_error.c_str() : runtime->error.c_str();
}

static const ggml_spec_accel_api API = {
    sizeof(ggml_spec_accel_api),
    GGML_SPEC_ACCEL_ABI_VERSION,
    api_query_capabilities,
    api_runtime_create,
    api_runtime_free,
    api_sequence_create,
    api_sequence_free,
    api_sequence_reset,
    api_sequence_invalidate,
    api_mtp_catchup,
    api_mtp_draft,
    api_dflash_features,
    api_dflash_draft,
    api_sequence_plan,
    api_sequence_remove_suffix,
    api_sequence_copy,
    api_sequence_shift,
    api_checkpoint_validate,
    api_checkpoint_commit,
    api_checkpoint_free,
    api_state_export,
    api_state_validate,
    api_state_import,
    api_state_cancel,
    api_blob_free,
    api_last_error,
};

} // namespace

extern "C" const ggml_spec_accel_api * ggml_backend_spec_accel_get_api(uint32_t abi_version) {
    return abi_version == GGML_SPEC_ACCEL_ABI_VERSION ? &API : nullptr;
}
