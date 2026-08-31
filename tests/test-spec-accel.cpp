#include "ggml-backend.h"
#include "ggml-spec-accel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t n_embd   = 5120;
constexpr uint32_t n_vocab  = 248320;
constexpr uint32_t n_head   = 24;
constexpr uint32_t n_head_kv = 4;
constexpr uint32_t head_dim = 256;
constexpr uint32_t n_rot    = 64;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int32_t device_ordinal(const char * name) {
    if (name == nullptr) {
        return 0;
    }
    const std::string value(name);
    size_t begin = value.size();
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(value[begin - 1]))) {
        --begin;
    }
    return begin == value.size() ? 0 : std::stoi(value.substr(begin));
}

struct fixture {
    const ggml_spec_accel_api * api = nullptr;
    ggml_spec_accel_runtime_t runtime = nullptr;
    std::vector<ggml_spec_accel_seq_t> sequences;

    ~fixture() {
        if (api == nullptr) {
            return;
        }
        for (auto sequence : sequences) {
            if (sequence != nullptr) {
                api->sequence_free(sequence);
            }
        }
        if (runtime != nullptr) {
            api->runtime_free(runtime);
        }
    }

    std::string error() const {
        if (api != nullptr && api->last_error != nullptr) {
            const char * value = api->last_error(runtime);
            if (value != nullptr && value[0] != '\0') {
                return value;
            }
        }
        return "no backend error was reported";
    }

    void ok(int32_t status, const char * operation) const {
        require(status == GGML_SPEC_ACCEL_STATUS_OK,
                std::string(operation) + " failed (status=" + std::to_string(status) + "): " + error());
    }
};

struct backend_binding {
    ggml_backend_dev_t device = nullptr;
    const ggml_spec_accel_api * api = nullptr;
};

backend_binding find_backend(const char * requested_backend) {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(device);
        if (requested_backend != nullptr && requested_backend[0] != '\0' &&
                (name == nullptr || std::strcmp(name, requested_backend) != 0)) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto getter = reg == nullptr ? nullptr : reinterpret_cast<ggml_backend_spec_accel_get_api_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_spec_accel_get_api"));
        const ggml_spec_accel_api * api = getter == nullptr ? nullptr :
                getter(GGML_SPEC_ACCEL_ABI_VERSION);
        if (api != nullptr && api->abi_version == GGML_SPEC_ACCEL_ABI_VERSION &&
                api->struct_size >= sizeof(*api)) {
            return {device, api};
        }
    }
    throw std::runtime_error("requested backend does not expose the speculative accelerator ABI");
}

ggml_spec_accel_sequence_plan plan(fixture & value, size_t sequence) {
    ggml_spec_accel_sequence_plan result {};
    result.struct_size = sizeof(result);
    value.ok(value.api->sequence_plan(value.sequences.at(sequence), &result), "sequence_plan");
    return result;
}

void catchup_tokens(fixture & value, size_t sequence, int32_t begin, uint32_t count) {
    std::vector<int32_t> tokens(count);
    std::vector<float> hidden(size_t(count) * n_embd, 0.0f);
    std::vector<int32_t> positions(size_t(count) * 4);
    for (uint32_t row = 0; row < count; ++row) {
        tokens[row] = int32_t(1 + row);
        for (uint32_t plane = 0; plane < 4; ++plane) {
            positions[size_t(plane) * count + row] = begin + int32_t(row);
        }
    }
    ggml_spec_accel_mtp_catchup input {};
    input.struct_size = sizeof(input);
    input.count = count;
    input.tokens = tokens.data();
    input.hidden_rows = hidden.data();
    input.positions = positions.data();
    input.hidden_width = n_embd;
    input.position_planes = 4;
    value.ok(value.api->mtp_catchup(value.sequences.at(sequence), &input), "text mtp_catchup");
}

void catchup_image(fixture & value, size_t sequence) {
    constexpr uint32_t count = 4;
    std::vector<float> embeddings(size_t(count) * n_embd, 0.0f);
    std::vector<float> hidden(size_t(count) * n_embd, 0.0f);
    // Four records share temporal position 8 while the spatial planes cover a
    // 2x2 grid.  Plane 3 follows Qwen's current image convention and remains 0.
    const std::array<int32_t, count * 4> positions = {
        8, 8, 8, 8,
        8, 8, 9, 9,
        8, 9, 8, 9,
        0, 0, 0, 0,
    };
    ggml_spec_accel_mtp_catchup input {};
    input.struct_size = sizeof(input);
    input.count = count;
    input.token_embeddings = embeddings.data();
    input.hidden_rows = hidden.data();
    input.positions = positions.data();
    input.token_embedding_width = n_embd;
    input.hidden_width = n_embd;
    input.position_planes = 4;
    value.ok(value.api->mtp_catchup(value.sequences.at(sequence), &input), "image mtp_catchup");
}

void test_runtime(
        const char * artifact,
        const char * target,
        const char * requested_backend,
        uint32_t cache_type) {
    const backend_binding binding = find_backend(requested_backend);
    ggml_backend_dev_t selected = binding.device;
    const ggml_spec_accel_api * api = binding.api;

    fixture value;
    value.api = api;
    ggml_spec_accel_descriptor descriptor {};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = GGML_SPEC_ACCEL_KIND_MTP;
    descriptor.artifact_path = artifact;
    descriptor.device_name = ggml_backend_dev_name(selected);
    descriptor.device_ordinal = device_ordinal(descriptor.device_name);
    descriptor.n_embd = n_embd;
    descriptor.n_vocab = n_vocab;
    descriptor.n_head = n_head;
    descriptor.n_head_kv = n_head_kv;
    descriptor.head_dim = head_dim;
    descriptor.n_rot = n_rot;
    descriptor.n_ctx = 128;
    descriptor.n_seq_max = 3;
    descriptor.cache_type = cache_type;
    descriptor.target_model_path = target;

    ggml_spec_accel_capabilities capabilities {};
    capabilities.struct_size = sizeof(capabilities);
    value.ok(api->query_capabilities(&descriptor, &capabilities), "query_capabilities");
    const uint64_t required = GGML_SPEC_ACCEL_CAP_MTP |
            GGML_SPEC_ACCEL_CAP_TOKEN_EMBEDDINGS |
            GGML_SPEC_ACCEL_CAP_MROPE4 |
            GGML_SPEC_ACCEL_CAP_MULTIMODAL_ROWS |
            GGML_SPEC_ACCEL_CAP_REMOVE_SUFFIX |
            GGML_SPEC_ACCEL_CAP_COPY_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_SHIFT_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_CHECKPOINT |
            GGML_SPEC_ACCEL_CAP_DURABLE_STATE;
    require((capabilities.flags & required) == required, "accelerator capability set is incomplete");

    ggml_spec_accel_descriptor unsupported_cache = descriptor;
    unsupported_cache.cache_type = 0xffffffffu;
    ggml_spec_accel_capabilities rejected {};
    rejected.struct_size = sizeof(rejected);
    require(api->query_capabilities(&unsupported_cache, &rejected) ==
                GGML_SPEC_ACCEL_STATUS_UNSUPPORTED,
            "unsupported private cache type did not fail closed");

    ggml_spec_accel_descriptor wrong_kind = descriptor;
    wrong_kind.kind = GGML_SPEC_ACCEL_KIND_DFLASH;
    wrong_kind.cache_type = GGML_SPEC_ACCEL_CACHE_F16;
    rejected = {};
    rejected.struct_size = sizeof(rejected);
    require(api->query_capabilities(&wrong_kind, &rejected) ==
                GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE,
            "MTP artifact was accepted as a DFlash artifact");

    value.ok(api->runtime_create(&descriptor, &value.runtime), "runtime_create");
    for (int32_t id = 0; id < 3; ++id) {
        ggml_spec_accel_seq_t sequence = nullptr;
        value.ok(api->sequence_create(value.runtime, id, &sequence), "sequence_create");
        value.sequences.push_back(sequence);
    }
    ggml_spec_accel_seq_t excess = nullptr;
    require(api->sequence_create(value.runtime, 3, &excess) == GGML_SPEC_ACCEL_STATUS_INVALID &&
                excess == nullptr,
            "runtime accepted more sequences than its declared slot capacity");

    catchup_tokens(value, 0, 0, 8);
    catchup_image(value, 0);
    auto multimodal = plan(value, 0);
    require(multimodal.record_count == 12 && multimodal.coverage_begin == 0 &&
            multimodal.coverage_end == 10 && multimodal.logical_pos == 10,
            "multimodal record/model-position accounting is incorrect");
    require(multimodal.can_shift == 0, "multimodal sequence unexpectedly advertised scalar shifting");
    require(api->sequence_shift(value.sequences[0], 0, 10, 1) == GGML_SPEC_ACCEL_STATUS_UNSUPPORTED,
            "multimodal sequence shift did not fail closed");

    std::vector<float> draft_hidden(n_embd, 0.0f);
    std::array<int32_t, 2> output {};
    ggml_spec_accel_mtp_draft draft {};
    draft.struct_size = sizeof(draft);
    draft.last_token = 9;
    draft.past_records = 12;
    draft.next_position = 10;
    draft.hidden = draft_hidden.data();
    draft.hidden_width = n_embd;
    draft.max_draft = output.size();
    draft.output_ids = output.data();
    draft.output_capacity = output.size();
    value.ok(api->mtp_draft(value.sequences[0], &draft), "mtp_draft");
    require(draft.output_count == output.size() &&
            std::all_of(output.begin(), output.end(), [](int32_t token) {
                return token >= 0 && uint32_t(token) < n_vocab;
            }), "MTP draft output is malformed");

    value.ok(api->sequence_remove_suffix(value.sequences[0], 10), "sequence_remove_suffix");
    multimodal = plan(value, 0);
    require(multimodal.record_count == 12 && multimodal.coverage_end == 10,
            "record-aware speculative rollback lost image rows");
    value.ok(api->sequence_copy(value.sequences[0], value.sequences[1], 0, 10), "sequence_copy");
    auto copied = plan(value, 1);
    require(copied.record_count == 12 && copied.coverage_end == 10,
            "multimodal fork did not copy every physical record");

    ggml_spec_accel_checkpoint_desc checkpoint {};
    checkpoint.struct_size = sizeof(checkpoint);
    checkpoint.logical_pos = copied.logical_pos;
    checkpoint.coverage_begin = copied.coverage_begin;
    checkpoint.coverage_end = copied.coverage_end;
    checkpoint.record_count = copied.record_count;
    checkpoint.generation = copied.generation;
    ggml_spec_accel_checkpoint_t transaction = nullptr;
    value.ok(api->checkpoint_validate(value.sequences[1], &checkpoint, &transaction),
            "checkpoint_validate");
    value.ok(api->checkpoint_commit(transaction), "checkpoint_commit");
    api->checkpoint_free(transaction);

    ggml_spec_accel_blob state {};
    state.struct_size = sizeof(state);
    value.ok(api->state_export(value.sequences[1], &state), "state_export");
    require(state.data != nullptr && state.size > 0, "durable state export was empty");
    value.ok(api->state_validate(value.sequences[2], state.data, state.size), "state_validate");
    value.ok(api->state_import(value.sequences[2], state.data, state.size), "state_import");
    const auto restored = plan(value, 2);
    require(restored.record_count == copied.record_count &&
            restored.coverage_begin == copied.coverage_begin &&
            restored.coverage_end == copied.coverage_end,
            "durable state restore changed sequence coverage");

    std::vector<uint8_t> malformed(state.size);
    std::memcpy(malformed.data(), state.data, state.size);
    malformed.back() ^= 0x5a;
    require(api->state_validate(value.sequences[2], malformed.data(), malformed.size()) ==
                GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE,
            "durable state checksum corruption was accepted");
    api->state_cancel(value.sequences[2]);
    api->blob_free(&state);

    value.ok(api->sequence_reset(value.sequences[1]), "sequence_reset");
    catchup_tokens(value, 1, 0, 16);
    value.ok(api->sequence_shift(value.sequences[1], 8, 16, -4), "sequence_shift");
    const auto shifted = plan(value, 1);
    require(shifted.record_count == 12 && shifted.coverage_begin == 0 &&
            shifted.coverage_end == 12 && shifted.logical_pos == 12,
            "packed MTP context shift produced incorrect coverage");

    // Slot 0 must remain untouched by slot 1's reset/shift transaction.
    const auto isolated = plan(value, 0);
    require(isolated.record_count == 12 && isolated.coverage_end == 10,
            "per-slot accelerator state is not isolated");
}

void dflash_features(
        fixture & value,
        size_t sequence,
        int32_t begin,
        uint32_t count,
        uint32_t feature_width) {
    std::vector<float> features(size_t(count) * feature_width, 0.0f);
    std::vector<int32_t> positions(size_t(count) * 4);
    for (uint32_t plane = 0; plane < 4; ++plane) {
        for (uint32_t row = 0; row < count; ++row) {
            positions[size_t(plane) * count + row] = begin + int32_t(row);
        }
    }
    ggml_spec_accel_dflash_features input {};
    input.struct_size = sizeof(input);
    input.count = count;
    input.features = features.data();
    input.feature_width = feature_width;
    input.positions = positions.data();
    input.position_planes = 4;
    value.ok(value.api->dflash_features(value.sequences.at(sequence), &input),
            "dflash_features");
}

void test_dflash_runtime(
        const char * artifact,
        const char * target,
        const char * requested_backend) {
    const backend_binding binding = find_backend(requested_backend);

    fixture value;
    value.api = binding.api;
    ggml_spec_accel_descriptor descriptor {};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = GGML_SPEC_ACCEL_KIND_DFLASH;
    descriptor.artifact_path = artifact;
    descriptor.device_name = ggml_backend_dev_name(binding.device);
    descriptor.device_ordinal = device_ordinal(descriptor.device_name);
    descriptor.n_embd = n_embd;
    descriptor.n_vocab = n_vocab;
    descriptor.n_head = n_head;
    descriptor.n_head_kv = n_head_kv;
    descriptor.head_dim = head_dim;
    descriptor.n_rot = n_rot;
    descriptor.n_ctx = 4096;
    descriptor.n_seq_max = 3;
    descriptor.cache_type = GGML_SPEC_ACCEL_CACHE_F16;
    descriptor.target_model_path = target;

    ggml_spec_accel_capabilities capabilities {};
    capabilities.struct_size = sizeof(capabilities);
    value.ok(value.api->query_capabilities(&descriptor, &capabilities),
            "DFlash query_capabilities");
    const uint64_t required = GGML_SPEC_ACCEL_CAP_DFLASH |
            GGML_SPEC_ACCEL_CAP_MROPE4 |
            GGML_SPEC_ACCEL_CAP_REMOVE_SUFFIX |
            GGML_SPEC_ACCEL_CAP_COPY_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_SHIFT_SEQUENCE |
            GGML_SPEC_ACCEL_CAP_CHECKPOINT |
            GGML_SPEC_ACCEL_CAP_DURABLE_STATE;
    require((capabilities.flags & required) == required &&
                capabilities.block_size == 8 && capabilities.cache_window == 2048 &&
                capabilities.feature_width == 25600 && capabilities.target_layer_count == 5,
            "DFlash capability descriptor is incomplete");

    value.ok(value.api->runtime_create(&descriptor, &value.runtime), "DFlash runtime_create");
    for (int32_t id = 0; id < 3; ++id) {
        ggml_spec_accel_seq_t sequence = nullptr;
        value.ok(value.api->sequence_create(value.runtime, id, &sequence),
                "DFlash sequence_create");
        value.sequences.push_back(sequence);
    }

    dflash_features(value, 0, 0, 8, capabilities.feature_width);
    auto source = plan(value, 0);
    require(source.logical_pos == 8 && source.coverage_begin == 0 &&
                source.coverage_end == 8 && source.record_count == 8,
            "DFlash feature injection produced incorrect coverage");

    std::array<int32_t, 7> output {};
    std::array<float, 7> confidence {};
    ggml_spec_accel_dflash_draft draft {};
    draft.struct_size = sizeof(draft);
    draft.last_token = 9;
    draft.past_tokens = 8;
    draft.max_draft = output.size();
    draft.output_ids = output.data();
    draft.output_capacity = output.size();
    draft.output_confidences = confidence.data();
    draft.confidence_capacity = confidence.size();
    value.ok(value.api->dflash_draft(value.sequences[0], &draft), "dflash_draft");
    require(draft.output_count > 0 && draft.output_count <= output.size(),
            "DFlash graph produced no candidates");
    for (uint32_t i = 0; i < draft.output_count; ++i) {
        require(output[i] >= 0 && uint32_t(output[i]) < n_vocab &&
                    std::isfinite(confidence[i]) && confidence[i] >= 0.0f && confidence[i] <= 1.0f,
                "DFlash selector output is malformed");
    }

    value.ok(value.api->sequence_copy(value.sequences[0], value.sequences[1], 0, 8),
            "DFlash sequence_copy");
    const auto copied = plan(value, 1);
    require(copied.coverage_begin == 0 && copied.coverage_end == 8,
            "DFlash fork coverage is incorrect");

    ggml_spec_accel_checkpoint_desc checkpoint {};
    checkpoint.struct_size = sizeof(checkpoint);
    checkpoint.logical_pos = copied.logical_pos;
    checkpoint.coverage_begin = copied.coverage_begin;
    checkpoint.coverage_end = copied.coverage_end;
    checkpoint.generation = copied.generation;
    ggml_spec_accel_checkpoint_t transaction = nullptr;
    value.ok(value.api->checkpoint_validate(value.sequences[1], &checkpoint, &transaction),
            "DFlash checkpoint_validate");
    value.ok(value.api->checkpoint_commit(transaction), "DFlash checkpoint_commit");
    value.api->checkpoint_free(transaction);

    ggml_spec_accel_blob state {};
    state.struct_size = sizeof(state);
    value.ok(value.api->state_export(value.sequences[1], &state), "DFlash state_export");
    require(state.data != nullptr && state.size > 0, "DFlash durable state export was empty");
    value.ok(value.api->state_validate(value.sequences[2], state.data, state.size),
            "DFlash state_validate");
    value.ok(value.api->state_import(value.sequences[2], state.data, state.size),
            "DFlash state_import");
    const auto restored = plan(value, 2);
    require(restored.coverage_begin == copied.coverage_begin &&
                restored.coverage_end == copied.coverage_end,
            "DFlash durable restore changed resident coverage");

    std::vector<uint8_t> malformed(state.size);
    std::memcpy(malformed.data(), state.data, state.size);
    malformed.back() ^= 0x5a;
    require(value.api->state_validate(value.sequences[2], malformed.data(), malformed.size()) ==
                GGML_SPEC_ACCEL_STATUS_INCOMPATIBLE,
            "corrupt DFlash durable state was accepted");
    value.api->state_cancel(value.sequences[2]);
    value.api->blob_free(&state);

    value.ok(value.api->sequence_remove_suffix(value.sequences[0], 6),
            "DFlash sequence_remove_suffix");
    source = plan(value, 0);
    require(source.coverage_end == 6, "DFlash suffix removal kept rejected rows");

    value.ok(value.api->sequence_reset(value.sequences[1]), "DFlash sequence_reset");
    dflash_features(value, 1, 0, 8, capabilities.feature_width);
    value.ok(value.api->sequence_shift(value.sequences[1], 0, 8, 1),
            "DFlash sequence_shift");
    const auto shifted = plan(value, 1);
    require(shifted.coverage_begin == 1 && shifted.coverage_end == 9 && shifted.logical_pos == 9,
            "DFlash delta-RoPE shift produced incorrect coverage");

    const auto isolated = plan(value, 2);
    require(isolated.coverage_begin == 0 && isolated.coverage_end == 8,
            "DFlash slot state is not isolated");
}

} // namespace

int main() {
    const char * artifact = std::getenv("GGML_SPEC_ACCEL_TEST_ARTIFACT");
    const char * target = std::getenv("GGML_SPEC_ACCEL_TEST_TARGET");
    if (artifact == nullptr || artifact[0] == '\0' || target == nullptr || target[0] == '\0') {
        std::printf("test-spec-accel: skipped (set GGML_SPEC_ACCEL_TEST_ARTIFACT and "
                    "GGML_SPEC_ACCEL_TEST_TARGET)\n");
        return 0;
    }

    const char * backend = std::getenv("GGML_SPEC_ACCEL_TEST_BACKEND");
    const char * cache = std::getenv("GGML_SPEC_ACCEL_TEST_CACHE");
    const char * dflash_artifact = std::getenv("GGML_SPEC_ACCEL_TEST_DFLASH_ARTIFACT");
    const uint32_t cache_type = cache != nullptr && std::strcmp(cache, "q8_0") == 0 ?
            GGML_SPEC_ACCEL_CACHE_Q8_0 : GGML_SPEC_ACCEL_CACHE_F16;
    try {
        test_runtime(artifact, target, backend, cache_type);
        std::printf("test-spec-accel: MTP %s lifecycle and multimodal parity OK\n",
                cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0 ? "q8_0" : "f16");
        if (dflash_artifact != nullptr && dflash_artifact[0] != '\0') {
            test_dflash_runtime(dflash_artifact, target, backend);
            std::printf("test-spec-accel: DFlash graph, selector, and lifecycle parity OK\n");
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-spec-accel: %s\n", error.what());
        return 1;
    }
}
