// Versioned optional speculative-accelerator ABI.
//
// This is deliberately a ggml backend interface rather than a llama/common
// interface.  Dynamic backends expose ggml_backend_spec_accel_get_api through
// ggml_backend_reg_get_proc_address(); callers never load a backend DLL
// directly and unsupported backends simply do not expose the entry point.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_SPEC_ACCEL_ABI_VERSION 3u

typedef void * ggml_spec_accel_runtime_t;
typedef void * ggml_spec_accel_seq_t;
typedef void * ggml_spec_accel_checkpoint_t;

enum ggml_spec_accel_status {
    GGML_SPEC_ACCEL_STATUS_OK             =  0,
    GGML_SPEC_ACCEL_STATUS_UNAVAILABLE    =  1,
    GGML_SPEC_ACCEL_STATUS_UNSUPPORTED    =  2,
    GGML_SPEC_ACCEL_STATUS_NO_COVERAGE    =  3,
    GGML_SPEC_ACCEL_STATUS_INVALID        = -1,
    GGML_SPEC_ACCEL_STATUS_IO             = -2,
    GGML_SPEC_ACCEL_STATUS_OUT_OF_MEMORY  = -3,
    GGML_SPEC_ACCEL_STATUS_RUNTIME        = -4,
    GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE   = -5,
};

enum ggml_spec_accel_kind {
    GGML_SPEC_ACCEL_KIND_MTP    = 1u,
    GGML_SPEC_ACCEL_KIND_DFLASH = 2u,
};

enum ggml_spec_accel_cache_type {
    GGML_SPEC_ACCEL_CACHE_F16  = 1u,
    GGML_SPEC_ACCEL_CACHE_Q8_0 = 2u,
};

enum ggml_spec_accel_capability {
    GGML_SPEC_ACCEL_CAP_MTP              = 1ull << 0,
    GGML_SPEC_ACCEL_CAP_DFLASH           = 1ull << 1,
    GGML_SPEC_ACCEL_CAP_TOKEN_EMBEDDINGS = 1ull << 2,
    GGML_SPEC_ACCEL_CAP_MROPE4           = 1ull << 3,
    GGML_SPEC_ACCEL_CAP_CACHE_F16        = 1ull << 4,
    GGML_SPEC_ACCEL_CAP_CACHE_Q8_0       = 1ull << 5,
    GGML_SPEC_ACCEL_CAP_REMOVE_SUFFIX    = 1ull << 6,
    GGML_SPEC_ACCEL_CAP_COPY_SEQUENCE    = 1ull << 7,
    GGML_SPEC_ACCEL_CAP_SHIFT_SEQUENCE   = 1ull << 8,
    GGML_SPEC_ACCEL_CAP_CHECKPOINT       = 1ull << 9,
    GGML_SPEC_ACCEL_CAP_DURABLE_STATE    = 1ull << 10,
    GGML_SPEC_ACCEL_CAP_ASYNC_ALLOC      = 1ull << 11,
    GGML_SPEC_ACCEL_CAP_MULTIMODAL_ROWS  = 1ull << 12,
};

struct ggml_spec_accel_descriptor {
    uint32_t struct_size;
    uint32_t kind;

    // The backend validates both the artifact fingerprint and these exact
    // model dimensions.  The initial ABI intentionally only admits a descriptor
    // that has been measured and validated by the backend.
    const char * artifact_path;
    const char * model_fingerprint;
    const char * device_name;
    int32_t device_ordinal;

    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t n_ctx;
    uint32_t n_seq_max;
    uint32_t cache_type;
    uint32_t flags;

    // Optional source model path used only for artifact identity validation.
    // Backends must not load target weights from this file for execution.
    const char * target_model_path;
};

struct ggml_spec_accel_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    uint32_t max_sequences;
    uint32_t max_context;
    uint32_t preferred_catchup_rows;
    uint32_t feature_width;
    uint32_t block_size;
    uint32_t cache_window;
    uint32_t target_layer_count;
    int32_t target_layers[8];
};

struct ggml_spec_accel_mtp_catchup {
    uint32_t struct_size;
    uint32_t count;

    // Exactly one of tokens or token_embeddings is present.  Hidden rows are
    // always FP32 target h_nextn rows.  positions contains four contiguous
    // planes of count entries, matching llama_batch M-RoPE layout.
    const int32_t * tokens;
    const float * token_embeddings;
    const float * hidden_rows;
    const int32_t * positions;
    uint32_t token_embedding_width;
    uint32_t hidden_width;
    uint32_t position_planes;
    uint32_t reserved;
};

struct ggml_spec_accel_mtp_draft {
    uint32_t struct_size;
    int32_t last_token;
    // Cache records and model positions are deliberately separate.  They are
    // equal for text-only input; an M-RoPE image can contribute many records
    // while advancing the temporal model position only by its grid extent.
    int32_t past_records;
    int32_t next_position;
    const float * hidden;
    uint32_t hidden_width;
    uint32_t max_draft;
    int32_t * output_ids;
    uint32_t output_capacity;
    uint32_t output_count;
};

struct ggml_spec_accel_dflash_features {
    uint32_t struct_size;
    uint32_t count;
    const float * features;
    uint32_t feature_width;
    const int32_t * positions;
    uint32_t position_planes;
};

struct ggml_spec_accel_dflash_draft {
    uint32_t struct_size;
    int32_t last_token;
    int32_t past_tokens;
    uint32_t max_draft;
    int32_t * output_ids;
    uint32_t output_capacity;
    uint32_t output_count;
    float * output_confidences;
    uint32_t confidence_capacity;
};

struct ggml_spec_accel_sequence_plan {
    uint32_t struct_size;
    int32_t logical_pos;
    int32_t coverage_begin;
    int32_t coverage_end;
    uint64_t generation;
    uint32_t can_remove_suffix;
    uint32_t can_shift;
    uint32_t record_count;
    uint32_t reserved;
};

struct ggml_spec_accel_checkpoint_desc {
    uint32_t struct_size;
    int32_t logical_pos;
    int32_t coverage_begin;
    int32_t coverage_end;
    uint32_t record_count;
    uint32_t reserved;
    uint64_t generation;
    const float * pending_hidden;
    uint32_t pending_hidden_count;
};

struct ggml_spec_accel_blob {
    uint32_t struct_size;
    void * data;
    size_t size;
};

struct ggml_spec_accel_api {
    uint32_t struct_size;
    uint32_t abi_version;

    int32_t (*query_capabilities)(
            const struct ggml_spec_accel_descriptor * descriptor,
            struct ggml_spec_accel_capabilities * capabilities);

    int32_t (*runtime_create)(
            const struct ggml_spec_accel_descriptor * descriptor,
            ggml_spec_accel_runtime_t * runtime);
    void (*runtime_free)(ggml_spec_accel_runtime_t runtime);

    int32_t (*sequence_create)(
            ggml_spec_accel_runtime_t runtime,
            int32_t sequence_id,
            ggml_spec_accel_seq_t * sequence);
    void (*sequence_free)(ggml_spec_accel_seq_t sequence);
    int32_t (*sequence_reset)(ggml_spec_accel_seq_t sequence);
    int32_t (*sequence_invalidate)(ggml_spec_accel_seq_t sequence, int32_t logical_pos);

    int32_t (*mtp_catchup)(
            ggml_spec_accel_seq_t sequence,
            const struct ggml_spec_accel_mtp_catchup * input);
    int32_t (*mtp_draft)(
            ggml_spec_accel_seq_t sequence,
            struct ggml_spec_accel_mtp_draft * input_output);

    int32_t (*dflash_features)(
            ggml_spec_accel_seq_t sequence,
            const struct ggml_spec_accel_dflash_features * input);
    int32_t (*dflash_draft)(
            ggml_spec_accel_seq_t sequence,
            struct ggml_spec_accel_dflash_draft * input_output);

    int32_t (*sequence_plan)(
            ggml_spec_accel_seq_t sequence,
            struct ggml_spec_accel_sequence_plan * plan);
    int32_t (*sequence_remove_suffix)(ggml_spec_accel_seq_t sequence, int32_t pos);
    int32_t (*sequence_copy)(
            ggml_spec_accel_seq_t source,
            ggml_spec_accel_seq_t destination,
            int32_t begin,
            int32_t end);
    int32_t (*sequence_shift)(
            ggml_spec_accel_seq_t sequence,
            int32_t begin,
            int32_t end,
            int32_t delta);

    int32_t (*checkpoint_validate)(
            ggml_spec_accel_seq_t sequence,
            const struct ggml_spec_accel_checkpoint_desc * checkpoint,
            ggml_spec_accel_checkpoint_t * transaction);
    int32_t (*checkpoint_commit)(ggml_spec_accel_checkpoint_t transaction);
    void (*checkpoint_free)(ggml_spec_accel_checkpoint_t transaction);

    int32_t (*state_export)(
            ggml_spec_accel_seq_t sequence,
            struct ggml_spec_accel_blob * state);
    // Validate identity, checksum, shape, and capacity without changing the
    // sequence's logical state.  This is the prepare phase used by atomic
    // server prompt-cache restoration.
    int32_t (*state_validate)(
            ggml_spec_accel_seq_t sequence,
            const void * data,
            size_t size);
    int32_t (*state_import)(
            ggml_spec_accel_seq_t sequence,
            const void * data,
            size_t size);
    // Release a state_validate staging allocation when a larger multi-part
    // restore transaction is abandoned before state_import.
    void (*state_cancel)(ggml_spec_accel_seq_t sequence);
    void (*blob_free)(struct ggml_spec_accel_blob * blob);

    const char * (*last_error)(ggml_spec_accel_runtime_t runtime);
};

typedef const struct ggml_spec_accel_api * (*ggml_backend_spec_accel_get_api_t)(uint32_t abi_version);

#ifdef __cplusplus
}
#endif
