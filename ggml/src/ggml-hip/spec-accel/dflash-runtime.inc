// SPDX-License-Identifier: MIT
// DFlash2 runtime integration.  Kernel math derives from BridgeSpec 0.1.0;
// artifact validation, per-slot ownership, ring coverage, and lifecycle are
// BeeLlama implementations.

struct dflash_expected_tensor {
    std::string name;
    ggml_type type;
    std::array<int64_t, 4> ne;
};

static std::vector<dflash_expected_tensor> dflash_schema() {
    std::vector<dflash_expected_tensor> schema = {
        {"enc.output_norm.weight",      GGML_TYPE_F32,  {5120, 1, 1, 1}},
        {"fc.weight",                   GGML_TYPE_Q4_K, {25600, 5120, 1, 1}},
        {"output_norm.weight",          GGML_TYPE_F32,  {5120, 1, 1, 1}},
        {"selector_hidden.weight",      GGML_TYPE_Q4_K, {5120, 256, 1, 1}},
        {"selector_predecessor.weight", GGML_TYPE_Q4_K, {256, 248320, 1, 1}},
        {"selector_successor.weight",   GGML_TYPE_Q4_K, {256, 248320, 1, 1}},
    };
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const ggml_type wide = layer == 2 || layer == 4 ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
        schema.insert(schema.end(), {
            {prefix + "attn_conv_base",        GGML_TYPE_F32,  {5120, 2, 2, 1}},
            {prefix + "attn_conv_proj.weight", GGML_TYPE_Q4_K, {5120, 1280, 1, 1}},
            {prefix + "attn_k.weight",         GGML_TYPE_Q4_K, {5120, 1024, 1, 1}},
            {prefix + "attn_k_norm.weight",    GGML_TYPE_F32,  {128, 1, 1, 1}},
            {prefix + "attn_norm.weight",      GGML_TYPE_F32,  {5120, 1, 1, 1}},
            {prefix + "attn_output.weight",    GGML_TYPE_Q4_K, {4096, 5120, 1, 1}},
            {prefix + "attn_q.weight",         GGML_TYPE_Q4_K, {5120, 4096, 1, 1}},
            {prefix + "attn_q_norm.weight",    GGML_TYPE_F32,  {128, 1, 1, 1}},
            {prefix + "attn_v.weight",         wide,           {5120, 1024, 1, 1}},
            {prefix + "ffn_conv_base",         GGML_TYPE_F32,  {5120, 2, 2, 1}},
            {prefix + "ffn_conv_proj.weight",  GGML_TYPE_Q4_K, {5120, 1280, 1, 1}},
            {prefix + "ffn_down.weight",       wide,           {17408, 5120, 1, 1}},
            {prefix + "ffn_gate.weight",       GGML_TYPE_Q4_K, {5120, 17408, 1, 1}},
            {prefix + "ffn_norm.weight",       GGML_TYPE_F32,  {5120, 1, 1, 1}},
            {prefix + "ffn_up.weight",         GGML_TYPE_Q4_K, {5120, 17408, 1, 1}},
        });
    }
    schema.push_back({"target.token_embd.weight", GGML_TYPE_Q4_0, {5120, 248320, 1, 1}});
    schema.push_back({"target.output.weight",     GGML_TYPE_Q6_K, {5120, 40960, 1, 1}});
    return schema;
}

static bool dflash_read_i32_array(
        const gguf_context * gguf,
        const char * key,
        int32_t * output,
        size_t count) {
    const int64_t id = key_id(gguf, key, GGUF_TYPE_ARRAY);
    if (id < 0 || gguf_get_arr_type(gguf, id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(gguf, id) != count) {
        return false;
    }
    std::memcpy(output, gguf_get_arr_data(gguf, id), count * sizeof(int32_t));
    return true;
}

static void dflash_dequant_q4k_host(const uint8_t * source, float * destination, size_t count) {
    const auto * blocks = reinterpret_cast<const block_q4_K *>(source);
    for (size_t block = 0; block < count / 256; ++block) {
        const block_q4_K * value = blocks + block;
        const float d = df_f16(value->d);
        const float dmin = df_f16(value->dmin);
        float * output = destination + block * 256;
        for (int group = 0; group < 4; ++group) {
            float scale0, minimum0, scale1, minimum1;
            q4k_scale_min(group * 2, value->scales, &scale0, &minimum0);
            q4k_scale_min(group * 2 + 1, value->scales, &scale1, &minimum1);
            const uint8_t * quant = value->qs + group * 32;
            for (int lane = 0; lane < 32; ++lane) {
                output[group * 64 + lane] =
                        d * scale0 * (quant[lane] & 0x0f) - dmin * minimum0;
                output[group * 64 + 32 + lane] =
                        d * scale1 * (quant[lane] >> 4) - dmin * minimum1;
            }
        }
    }
}

static bool query_dflash_artifact_descriptor(
        const char * path,
        ggml_spec_accel_capabilities * capabilities,
        std::string & error) {
    if (path == nullptr || path[0] == '\0') {
        error = "a DFlash accelerator artifact is required";
        return false;
    }
    gguf_init_params params {};
    params.no_alloc = true;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_file(path, params), gguf_free);
    if (!gguf) {
        error = std::string("cannot read DFlash accelerator GGUF: ") + path;
        return false;
    }
    const int64_t kind_id = key_id(gguf.get(), "beellama.spec_accel.kind", GGUF_TYPE_STRING);
    const int64_t version_id = key_id(gguf.get(), "beellama.spec_accel.version", GGUF_TYPE_UINT32);
    const int64_t window_id = key_id(gguf.get(), "beellama.spec_accel.sliding_window", GGUF_TYPE_UINT32);
    const int64_t mask_id = key_id(gguf.get(), "beellama.spec_accel.mask_token_id", GGUF_TYPE_INT32);
    if (kind_id < 0 || std::strcmp(gguf_get_val_str(gguf.get(), kind_id), "dflash") != 0 ||
            version_id < 0 || gguf_get_val_u32(gguf.get(), version_id) != SPEC_ARTIFACT_VERSION ||
            window_id < 0 || gguf_get_val_u32(gguf.get(), window_id) != DF_WINDOW ||
            mask_id < 0 || gguf_get_val_i32(gguf.get(), mask_id) < 0 ||
            gguf_get_val_i32(gguf.get(), mask_id) >= SPEC_VOCAB ||
            !dflash_read_i32_array(gguf.get(), "beellama.spec_accel.target_layers",
                capabilities->target_layers, 5)) {
        error = "DFlash accelerator descriptor is missing or incompatible";
        return false;
    }
    capabilities->feature_width = 25600;
    capabilities->block_size = DF_BLOCK;
    capabilities->cache_window = DF_WINDOW;
    capabilities->target_layer_count = 5;
    return true;
}

static bool load_dflash_artifact(spec_runtime * runtime) {
    gguf_init_params params {};
    params.no_alloc = true;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
            gguf_init_from_file(runtime->artifact_path.c_str(), params), gguf_free);
    if (!gguf) {
        runtime->error = "cannot read DFlash accelerator GGUF: " + runtime->artifact_path;
        return false;
    }
    const int64_t arch_id = key_id(gguf.get(), "general.architecture", GGUF_TYPE_STRING);
    const int64_t version_id = key_id(gguf.get(), "beellama.spec_accel.version", GGUF_TYPE_UINT32);
    const int64_t kind_id = key_id(gguf.get(), "beellama.spec_accel.kind", GGUF_TYPE_STRING);
    const int64_t fingerprint_id = key_id(
            gguf.get(), "beellama.spec_accel.target_fingerprint", GGUF_TYPE_STRING);
    const int64_t ids_id = key_id(gguf.get(), "beellama.spec_accel.draft_vocab_ids", GGUF_TYPE_ARRAY);
    const int64_t frequency_id = key_id(
            gguf.get(), "beellama.spec_accel.rope_freq_base", GGUF_TYPE_FLOAT32);
    const int64_t window_id = key_id(
            gguf.get(), "beellama.spec_accel.sliding_window", GGUF_TYPE_UINT32);
    const int64_t mask_id = key_id(
            gguf.get(), "beellama.spec_accel.mask_token_id", GGUF_TYPE_INT32);
    if (arch_id < 0 || std::strcmp(gguf_get_val_str(gguf.get(), arch_id), "beellama-spec-accel") != 0 ||
            version_id < 0 || gguf_get_val_u32(gguf.get(), version_id) != SPEC_ARTIFACT_VERSION ||
            kind_id < 0 || std::strcmp(gguf_get_val_str(gguf.get(), kind_id), "dflash") != 0 ||
            fingerprint_id < 0 || ids_id < 0 || frequency_id < 0 || window_id < 0 || mask_id < 0 ||
            gguf_get_val_u32(gguf.get(), window_id) != DF_WINDOW ||
            gguf_get_arr_type(gguf.get(), ids_id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(gguf.get(), ids_id) != 40960) {
        runtime->error = "DFlash accelerator metadata is missing, malformed, or incompatible";
        return false;
    }

    std::string sampled;
    if (!sample_target_fingerprint(runtime->target_path.c_str(), sampled, runtime->error)) {
        return false;
    }
    const std::string artifact_fingerprint = gguf_get_val_str(gguf.get(), fingerprint_id);
    if (sampled != artifact_fingerprint ||
            (!runtime->model_fingerprint.empty() && runtime->model_fingerprint != artifact_fingerprint)) {
        runtime->error = "DFlash artifact was prepared from a different target model";
        return false;
    }
    runtime->model_fingerprint = artifact_fingerprint;

    auto & weights = runtime->dflash;
    if (!dflash_read_i32_array(gguf.get(), "beellama.spec_accel.target_layers",
                weights.target_layers.data(), weights.target_layers.size()) ||
            !dflash_read_i32_array(gguf.get(), "beellama.spec_accel.mrope_sections",
                weights.mrope_sections.data(), weights.mrope_sections.size())) {
        runtime->error = "DFlash target-layer or M-RoPE descriptor is invalid";
        return false;
    }
    int sections = 0;
    for (int32_t value : weights.mrope_sections) {
        if (value < 0) {
            runtime->error = "DFlash M-RoPE descriptor contains a negative section";
            return false;
        }
        sections += value;
    }
    if (sections != DF_HDIM / 2) {
        runtime->error = "DFlash M-RoPE sections do not cover the rotary dimension";
        return false;
    }
    weights.rope_freq_base = gguf_get_val_f32(gguf.get(), frequency_id);
    if (!(weights.rope_freq_base > 0.0f)) {
        runtime->error = "DFlash RoPE frequency base is invalid";
        return false;
    }
    weights.mask_token_id = gguf_get_val_i32(gguf.get(), mask_id);
    if (weights.mask_token_id < 0 || weights.mask_token_id >= SPEC_VOCAB) {
        runtime->error = "DFlash mask token descriptor is invalid";
        return false;
    }

    const auto schema = dflash_schema();
    if (gguf_get_n_tensors(gguf.get()) != int64_t(schema.size())) {
        runtime->error = "DFlash accelerator GGUF must contain exactly 83 tensors";
        return false;
    }
    for (const auto & expected : schema) {
        const int64_t id = gguf_find_tensor(gguf.get(), expected.name.c_str());
        if (id < 0 || gguf_get_tensor_type(gguf.get(), id) != expected.type) {
            runtime->error = "missing or incompatible DFlash tensor: " + expected.name;
            return false;
        }
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), id);
        for (size_t dim = 0; dim < expected.ne.size(); ++dim) {
            if (ne[dim] != expected.ne[dim]) {
                runtime->error = "DFlash tensor shape mismatch: " + expected.name;
                return false;
            }
        }
    }

    weights.remap.resize(40960);
    std::memcpy(weights.remap.data(), gguf_get_arr_data(gguf.get(), ids_id),
            weights.remap.size() * sizeof(int32_t));
    std::vector<uint8_t> seen(SPEC_VOCAB, 0);
    for (int32_t token : weights.remap) {
        if (token < 0 || token >= SPEC_VOCAB || seen[token] != 0) {
            runtime->error = "DFlash vocabulary remap is invalid or non-injective";
            return false;
        }
        seen[token] = 1;
    }

    FILE * file = std::fopen(runtime->artifact_path.c_str(), "rb");
    if (file == nullptr) {
        runtime->error = "cannot open DFlash accelerator tensor data";
        return false;
    }
    for (const auto & expected : schema) {
        const int64_t id = gguf_find_tensor(gguf.get(), expected.name.c_str());
        if (expected.name == "selector_predecessor.weight" ||
                expected.name == "selector_successor.weight") {
            const size_t size = gguf_get_tensor_size(gguf.get(), id);
            std::vector<uint8_t> raw;
            try {
                raw.resize(size);
                std::vector<float> & table = expected.name == "selector_predecessor.weight" ?
                        weights.selector_prev : weights.selector_next;
                table.resize(size_t(SPEC_VOCAB) * DF_RANK);
                if (!read_exact(file,
                            gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), id),
                            raw.data(), raw.size())) {
                    std::fclose(file);
                    runtime->error = "short read while loading " + expected.name;
                    return false;
                }
                dflash_dequant_q4k_host(raw.data(), table.data(), table.size());
            } catch (const std::bad_alloc &) {
                std::fclose(file);
                runtime->error = "host allocation failed while loading DFlash selector tables";
                return false;
            }
            continue;
        }
        if (!upload_tensor(runtime, file, gguf.get(), id,
                    expected.name == "target.token_embd.weight")) {
            std::fclose(file);
            return false;
        }
    }
    std::fclose(file);

    const auto q4 = [&](const std::string & name) {
        return static_cast<const block_q4_K *>(required_tensor(runtime, name));
    };
    const auto f32 = [&](const std::string & name) {
        return static_cast<const float *>(required_tensor(runtime, name));
    };
    weights.fc = q4("fc.weight");
    weights.selector_hidden = q4("selector_hidden.weight");
    weights.target_embedding = static_cast<const block_q4_0d *>(
            required_tensor(runtime, "target.token_embd.weight"));
    weights.target_head = static_cast<const block_q6_K *>(required_tensor(runtime, "target.output.weight"));
    weights.enc_norm = f32("enc.output_norm.weight");
    weights.output_norm = f32("output_norm.weight");
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        auto & value = weights.layers[layer];
        value.conv_a_proj = q4(prefix + "attn_conv_proj.weight");
        value.conv_f_proj = q4(prefix + "ffn_conv_proj.weight");
        value.wq = q4(prefix + "attn_q.weight");
        value.wk = q4(prefix + "attn_k.weight");
        value.wo = q4(prefix + "attn_output.weight");
        value.up = q4(prefix + "ffn_up.weight");
        value.gate = q4(prefix + "ffn_gate.weight");
        value.wv = required_tensor(runtime, prefix + "attn_v.weight");
        value.down = required_tensor(runtime, prefix + "ffn_down.weight");
        value.wv_q6 = layer == 2 || layer == 4;
        value.down_q6 = value.wv_q6;
        value.attn_norm = f32(prefix + "attn_norm.weight");
        value.qnorm = f32(prefix + "attn_q_norm.weight");
        value.knorm = f32(prefix + "attn_k_norm.weight");
        value.ffn_norm = f32(prefix + "ffn_norm.weight");
        value.conv_a_base = f32(prefix + "attn_conv_base");
        value.conv_f_base = f32(prefix + "ffn_conv_base");
    }
    return true;
}

static void dflash_gemv_q4(
        int columns,
        const block_q4_K * weights,
        const float * input,
        float * output,
        int n_in,
        int n_out,
        hipStream_t stream) {
    const dim3 grid((n_out + 7) / 8);
    switch (columns) {
        case 1: gemv8t_q4_K<1><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 2: gemv8t_q4_K<2><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 3: gemv8t_q4_K<3><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 4: gemv8t_q4_K<4><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 5: gemv8t_q4_K<5><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 6: gemv8t_q4_K<6><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 7: gemv8t_q4_K<7><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        default: gemv8t_q4_K<8><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
    }
}

static void dflash_gemv_q6(
        int columns,
        const block_q6_K * weights,
        const float * input,
        float * output,
        int n_in,
        int n_out,
        hipStream_t stream) {
    const dim3 grid((n_out + 7) / 8);
    switch (columns) {
        case 1: gemv8t_q6_K<1><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 2: gemv8t_q6_K<2><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 3: gemv8t_q6_K<3><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 4: gemv8t_q6_K<4><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 5: gemv8t_q6_K<5><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 6: gemv8t_q6_K<6><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        case 7: gemv8t_q6_K<7><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
        default: gemv8t_q6_K<8><<<grid, 256, 0, stream>>>(weights, input, output, n_in, n_out); break;
    }
}

static void free_dflash_batch_storage(spec_sequence * sequence) {
    auto & batch = sequence->dflash_batch;
    void * pointers[] = {
        batch.features, batch.encoded, batch.xn, batch.k, batch.v,
        batch.positions, batch.gemm_input,
    };
    for (void * pointer : pointers) {
        device_free(sequence->runtime, pointer, sequence->stream);
    }
    batch = {};
}

static bool ensure_dflash_batch_capacity(spec_sequence * sequence, uint32_t capacity) {
    auto & batch = sequence->dflash_batch;
    if (capacity <= batch.capacity) {
        return true;
    }
    free_dflash_batch_storage(sequence);

    batch.capacity = capacity;
    const size_t rows = capacity;
    const bool ok =
        alloc_array(sequence, batch.features, rows * 25600, "batched DFlash target features") &&
        alloc_array(sequence, batch.encoded, rows * DF_EMBD, "batched DFlash encoded rows") &&
        alloc_array(sequence, batch.xn, rows * DF_EMBD, "batched DFlash normalized rows") &&
        alloc_array(sequence, batch.k, rows * DF_KV * DF_HDIM, "batched DFlash K") &&
        alloc_array(sequence, batch.v, rows * DF_KV * DF_HDIM, "batched DFlash V") &&
        alloc_array(sequence, batch.positions, rows * 4, "batched DFlash M-RoPE positions") &&
        alloc_array(sequence, batch.gemm_input, rows * 25600, "batched DFlash GEMM input");
    if (!ok) {
        free_dflash_batch_storage(sequence);
        return false;
    }
    return true;
}

static bool ensure_dflash_dense_weights(spec_runtime * runtime) {
    std::lock_guard<std::mutex> lock(runtime->dense_mutex);
    auto & dense = runtime->dflash_dense;
    if (dense.ready) {
        return true;
    }
    if (dense.unavailable) {
        return false;
    }

    constexpr size_t fc_elements = size_t(DF_EMBD) * 25600;
    constexpr size_t kv_elements = size_t(DF_KV) * DF_HDIM * DF_EMBD;
    dense.fc = static_cast<__half *>(device_alloc(
            runtime, fc_elements * sizeof(__half), runtime->upload_stream));
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        dense.k[layer] = static_cast<__half *>(device_alloc(
                runtime, kv_elements * sizeof(__half), runtime->upload_stream));
        dense.v[layer] = static_cast<__half *>(device_alloc(
                runtime, kv_elements * sizeof(__half), runtime->upload_stream));
    }
    const auto cleanup = [&]() {
        device_free(runtime, dense.fc, runtime->upload_stream);
        for (int layer = 0; layer < DF_LAYERS; ++layer) {
            device_free(runtime, dense.k[layer], runtime->upload_stream);
            device_free(runtime, dense.v[layer], runtime->upload_stream);
        }
        hipStreamSynchronize(runtime->upload_stream);
        dense = {};
        dense.unavailable = true;
        // This route is an optimization.  Resource pressure must retain the
        // existing quantized width-8 feature path rather than disable DFlash.
        runtime->error.clear();
        global_error.clear();
    };
    if (dense.fc == nullptr) {
        cleanup();
        return false;
    }
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        if (dense.k[layer] == nullptr || dense.v[layer] == nullptr) {
            cleanup();
            return false;
        }
    }

    constexpr int threads = 256;
    df_q4k_matrix_to_f16<<<(fc_elements + threads - 1) / threads, threads,
            0, runtime->upload_stream>>>(runtime->dflash.fc, dense.fc, fc_elements, 25600);
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        const auto & source = runtime->dflash.layers[layer];
        df_q4k_matrix_to_f16<<<(kv_elements + threads - 1) / threads, threads,
                0, runtime->upload_stream>>>(source.wk, dense.k[layer], kv_elements, DF_EMBD);
        if (source.wv_q6) {
            df_q6k_matrix_to_f16<<<(kv_elements + threads - 1) / threads, threads,
                    0, runtime->upload_stream>>>(
                    static_cast<const block_q6_K *>(source.wv), dense.v[layer], kv_elements, DF_EMBD);
        } else {
            df_q4k_matrix_to_f16<<<(kv_elements + threads - 1) / threads, threads,
                    0, runtime->upload_stream>>>(
                    static_cast<const block_q4_K *>(source.wv), dense.v[layer], kv_elements, DF_EMBD);
        }
    }
    if (!hip_ok(runtime, hipGetLastError(), "materialize DFlash FP16 projections") ||
            !hip_ok(runtime, hipStreamSynchronize(runtime->upload_stream),
                "synchronize DFlash projection materialization")) {
        cleanup();
        return false;
    }

    dense.ready = true;
    runtime->allocations.push_back({"dense.dflash.fc.f16", dense.fc, nullptr,
            fc_elements * sizeof(__half)});
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        runtime->allocations.push_back({"dense.dflash." + std::to_string(layer) + ".k.f16",
                dense.k[layer], nullptr, kv_elements * sizeof(__half)});
        runtime->allocations.push_back({"dense.dflash." + std::to_string(layer) + ".v.f16",
                dense.v[layer], nullptr, kv_elements * sizeof(__half)});
    }
    return true;
}

static bool dflash_dense_gemm(
        spec_sequence * sequence,
        const __half * weights,
        const float * input,
        float * output,
        int n_in,
        int n_out,
        int rows) {
    auto * runtime = sequence->runtime;
    auto & batch = sequence->dflash_batch;
    if (sequence->blas == nullptr || batch.gemm_input == nullptr || weights == nullptr) {
        runtime->error = "DFlash dense feature route is not initialized";
        return false;
    }
    const size_t elements = size_t(rows) * n_in;
    constexpr int threads = 256;
    f32_matrix_to_f16<<<(elements + threads - 1) / threads, threads, 0, sequence->stream>>>(
            input, batch.gemm_input, elements);
    if (!hip_ok(runtime, hipGetLastError(), "convert DFlash feature input to FP16")) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
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
            "run batched DFlash FP16 GEMM");
}

static void free_dflash_storage(spec_sequence * sequence) {
    if (sequence == nullptr) {
        return;
    }
    auto & value = sequence->dflash;
    if (value.graph_exec != nullptr) {
        hipGraphExecDestroy(value.graph_exec);
        value.graph_exec = nullptr;
    }
    void * pointers[] = {
        value.features, value.encoded, value.x, value.xn, value.dyn_a, value.dyn_f,
        value.conv, value.q, value.k, value.v, value.attention, value.output,
        value.ffn_up, value.ffn_gate, value.ffn_tmp, value.token_embedding,
        value.logits, value.selector_hidden, value.topk_part_max, value.topk_values,
        value.positions, value.tokens, value.topk_part_ids, value.topk_ids,
    };
    for (void * pointer : pointers) {
        device_free(sequence->runtime, pointer, sequence->stream);
    }
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        device_free(sequence->runtime, value.k_cache[layer], sequence->stream);
        device_free(sequence->runtime, value.v_cache[layer], sequence->stream);
    }
    value = {};
    free_dflash_batch_storage(sequence);
}

static bool allocate_dflash_storage(spec_sequence * sequence) {
    auto & value = sequence->dflash;
    value.topk_parts = (40960 + 2047) / 2048;
    const bool ok =
        alloc_array(sequence, value.features, size_t(DF_BLOCK) * 25600, "DFlash feature rows") &&
        alloc_array(sequence, value.encoded, size_t(DF_BLOCK) * DF_EMBD, "DFlash encoded rows") &&
        alloc_array(sequence, value.x, size_t(DF_BLOCK) * DF_EMBD, "DFlash hidden rows") &&
        alloc_array(sequence, value.xn, size_t(DF_BLOCK) * DF_EMBD, "DFlash normalized rows") &&
        alloc_array(sequence, value.dyn_a, size_t(DF_BLOCK) * DF_CONVDYN, "DFlash attention convolution") &&
        alloc_array(sequence, value.dyn_f, size_t(DF_BLOCK) * DF_CONVDYN, "DFlash FFN convolution") &&
        alloc_array(sequence, value.conv, size_t(DF_BLOCK) * DF_EMBD, "DFlash convolution output") &&
        alloc_array(sequence, value.q, size_t(DF_BLOCK) * DF_HEADS * DF_HDIM, "DFlash Q") &&
        alloc_array(sequence, value.k, size_t(DF_BLOCK) * DF_KV * DF_HDIM, "DFlash K") &&
        alloc_array(sequence, value.v, size_t(DF_BLOCK) * DF_KV * DF_HDIM, "DFlash V") &&
        alloc_array(sequence, value.attention, size_t(DF_BLOCK) * DF_HEADS * DF_HDIM, "DFlash attention output") &&
        alloc_array(sequence, value.output, size_t(DF_BLOCK) * DF_EMBD, "DFlash projection output") &&
        alloc_array(sequence, value.ffn_up, size_t(DF_BLOCK) * DF_FF, "DFlash FFN up") &&
        alloc_array(sequence, value.ffn_gate, size_t(DF_BLOCK) * DF_FF, "DFlash FFN gate") &&
        alloc_array(sequence, value.ffn_tmp, size_t(DF_BLOCK) * DF_EMBD, "DFlash FFN output") &&
        alloc_array(sequence, value.token_embedding, size_t(DF_BLOCK) * DF_EMBD, "DFlash normalized output") &&
        alloc_array(sequence, value.logits, size_t(DF_BLOCK) * 40960, "DFlash sliced logits") &&
        alloc_array(sequence, value.selector_hidden, size_t(DF_BLOCK) * DF_RANK, "DFlash selector hidden") &&
        alloc_array(sequence, value.positions, size_t(DF_BLOCK) * 4, "DFlash M-RoPE positions") &&
        alloc_array(sequence, value.tokens, DF_BLOCK, "DFlash tokens") &&
        alloc_array(sequence, value.topk_part_max, size_t(DF_BLOCK) * value.topk_parts, "DFlash top-k maxima") &&
        alloc_array(sequence, value.topk_part_ids, size_t(DF_BLOCK) * value.topk_parts, "DFlash top-k indices") &&
        alloc_array(sequence, value.topk_ids, size_t(DF_BLOCK) * DF_TOPK, "DFlash top-k IDs") &&
        alloc_array(sequence, value.topk_values, size_t(DF_BLOCK) * DF_TOPK, "DFlash top-k values");
    if (!ok) {
        free_dflash_storage(sequence);
        return false;
    }
    constexpr size_t cache_values = size_t(DF_WINDOW) * DF_KV * DF_HDIM;
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        if (!alloc_array(sequence, value.k_cache[layer], cache_values, "DFlash K ring") ||
                !alloc_array(sequence, value.v_cache[layer], cache_values, "DFlash V ring")) {
            free_dflash_storage(sequence);
            return false;
        }
    }
    value.feature_capacity = DF_BLOCK;
    return true;
}

static constexpr size_t dflash_cache_row_bytes() {
    return size_t(DF_KV) * DF_HDIM * sizeof(__half);
}

// DFlash stores only a bounded logical window in a modulo-addressed ring.
// Lifecycle operations always move logical ranges so stale physical rows can
// never be mistaken for resident coverage after a fork, shift, or restore.
static bool dflash_ring_to_contiguous(
        spec_runtime * runtime,
        const __half * ring,
        int32_t logical_begin,
        int32_t count,
        void * contiguous,
        hipMemcpyKind kind,
        hipStream_t stream,
        const char * operation) {
    if (logical_begin < 0 || count < 0 || count > DF_WINDOW ||
            (count > 0 && (ring == nullptr || contiguous == nullptr))) {
        runtime->error = "invalid DFlash ring read";
        return false;
    }
    if (count == 0) {
        return true;
    }
    const size_t row_bytes = dflash_cache_row_bytes();
    const int32_t slot = logical_begin % DF_WINDOW;
    const int32_t first = std::min(count, DF_WINDOW - slot);
    auto * destination = static_cast<uint8_t *>(contiguous);
    if (!hip_ok(runtime, hipMemcpyAsync(destination, ring + size_t(slot) * DF_KV * DF_HDIM,
                    size_t(first) * row_bytes, kind, stream), operation)) {
        return false;
    }
    const int32_t second = count - first;
    return second == 0 || hip_ok(runtime, hipMemcpyAsync(
            destination + size_t(first) * row_bytes, ring,
            size_t(second) * row_bytes, kind, stream), operation);
}

static bool dflash_contiguous_to_ring(
        spec_runtime * runtime,
        const void * contiguous,
        int32_t logical_begin,
        int32_t count,
        __half * ring,
        hipMemcpyKind kind,
        hipStream_t stream,
        const char * operation) {
    if (logical_begin < 0 || count < 0 || count > DF_WINDOW ||
            (count > 0 && (ring == nullptr || contiguous == nullptr))) {
        runtime->error = "invalid DFlash ring write";
        return false;
    }
    if (count == 0) {
        return true;
    }
    const size_t row_bytes = dflash_cache_row_bytes();
    const int32_t slot = logical_begin % DF_WINDOW;
    const int32_t first = std::min(count, DF_WINDOW - slot);
    const auto * source = static_cast<const uint8_t *>(contiguous);
    if (!hip_ok(runtime, hipMemcpyAsync(ring + size_t(slot) * DF_KV * DF_HDIM, source,
                    size_t(first) * row_bytes, kind, stream), operation)) {
        return false;
    }
    const int32_t second = count - first;
    return second == 0 || hip_ok(runtime, hipMemcpyAsync(
            ring, source + size_t(first) * row_bytes,
            size_t(second) * row_bytes, kind, stream), operation);
}

static bool launch_dflash_features(
        spec_sequence * sequence,
        const ggml_spec_accel_dflash_features * input) {
    auto * runtime = sequence->runtime;
    auto & scratch = sequence->dflash;
    auto & weights = runtime->dflash;
    if (input->count == 0) {
        return true;
    }
    if (input->features == nullptr || input->positions == nullptr ||
            input->feature_width != 25600 || input->position_planes != 4) {
        runtime->error = "invalid DFlash feature input";
        return false;
    }

    int32_t previous = -1;
    bool multimodal = false;
    for (uint32_t row = 0; row < input->count; ++row) {
        const int32_t position = input->positions[row];
        if (position < 0 || uint32_t(position) >= runtime->descriptor.n_ctx ||
                (previous >= 0 && position != previous + 1)) {
            runtime->error = "DFlash feature positions must be contiguous and inside the target context";
            return false;
        }
        previous = position;
        for (uint32_t plane = 1; plane < 4; ++plane) {
            multimodal = multimodal ||
                    input->positions[size_t(plane) * input->count + row] != position;
        }
    }

    const int32_t first = input->positions[0];
    if (sequence->coverage_end != first) {
        sequence->coverage_begin = first;
        sequence->coverage_end = first;
    }

    constexpr uint32_t dense_min_rows = 32;
    uint32_t tile_limit = DF_BLOCK;
    bool use_dense = false;
    if (input->count >= dense_min_rows && sequence->blas != nullptr &&
            ensure_dflash_dense_weights(runtime)) {
        size_t free_vram = 0;
        size_t total_vram = 0;
        constexpr size_t scratch_bytes_per_row =
                size_t(25600 + DF_EMBD + DF_EMBD + DF_KV * DF_HDIM * 2) * sizeof(float) +
                size_t(25600) * sizeof(__half) + 4 * sizeof(int32_t);
        if (hipMemGetInfo(&free_vram, &total_vram) == hipSuccess) {
            const size_t rows_by_budget = (free_vram / 8) / scratch_bytes_per_row;
            tile_limit = uint32_t(std::min<size_t>(input->count,
                    std::min<size_t>(rows_by_budget, std::numeric_limits<uint32_t>::max())));
            if (tile_limit >= dense_min_rows && ensure_dflash_batch_capacity(sequence, tile_limit)) {
                use_dense = true;
            }
        } else {
            hipGetLastError();
        }
        if (!use_dense) {
            // A dense prefill allocation is optional.  Preserve the already
            // validated width-8 path under low-VRAM or allocator pressure.
            runtime->error.clear();
            global_error.clear();
            tile_limit = DF_BLOCK;
        }
    }

    for (uint32_t offset = 0; offset < input->count; offset += tile_limit) {
        const int columns = int(std::min<uint32_t>(tile_limit, input->count - offset));
        std::vector<int32_t> positions(size_t(columns) * 4);
        for (int plane = 0; plane < 4; ++plane) {
            for (int row = 0; row < columns; ++row) {
                positions[size_t(plane) * columns + row] =
                        input->positions[size_t(plane) * input->count + offset + row];
            }
        }

        float * device_features = use_dense ? sequence->dflash_batch.features : scratch.features;
        float * encoded = use_dense ? sequence->dflash_batch.encoded : scratch.encoded;
        float * normalized = use_dense ? sequence->dflash_batch.xn : scratch.xn;
        float * projected_k = use_dense ? sequence->dflash_batch.k : scratch.k;
        float * projected_v = use_dense ? sequence->dflash_batch.v : scratch.v;
        int32_t * device_positions = use_dense ? sequence->dflash_batch.positions : scratch.positions;

        if (!hip_ok(runtime, hipMemcpyAsync(device_features,
                        input->features + size_t(offset) * 25600,
                        size_t(columns) * 25600 * sizeof(float), hipMemcpyHostToDevice, sequence->stream),
                    "upload DFlash target features") ||
                !hip_ok(runtime, hipMemcpyAsync(device_positions, positions.data(),
                        size_t(columns) * 4 * sizeof(int32_t), hipMemcpyHostToDevice, sequence->stream),
                    "upload DFlash feature positions")) {
            return false;
        }

        if (use_dense) {
            if (!dflash_dense_gemm(sequence, runtime->dflash_dense.fc,
                        device_features, encoded, 25600, DF_EMBD, columns)) {
                return false;
            }
        } else {
            dflash_gemv_q4(columns, weights.fc, device_features, encoded,
                    25600, DF_EMBD, sequence->stream);
        }
        df_rmsnorm<<<columns, 256, 0, sequence->stream>>>(
                encoded, weights.enc_norm, normalized, DF_EMBD);
        for (int layer = 0; layer < DF_LAYERS; ++layer) {
            auto & value = weights.layers[layer];
            if (use_dense) {
                if (!dflash_dense_gemm(sequence, runtime->dflash_dense.k[layer],
                            normalized, projected_k, DF_EMBD, DF_KV * DF_HDIM, columns) ||
                        !dflash_dense_gemm(sequence, runtime->dflash_dense.v[layer],
                            normalized, projected_v, DF_EMBD, DF_KV * DF_HDIM, columns)) {
                    return false;
                }
            } else {
                dflash_gemv_q4(columns, value.wk, normalized, projected_k,
                        DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
                if (value.wv_q6) {
                    dflash_gemv_q6(columns, static_cast<const block_q6_K *>(value.wv), normalized,
                            projected_v, DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
                } else {
                    dflash_gemv_q4(columns, static_cast<const block_q4_K *>(value.wv), normalized,
                            projected_v, DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
                }
            }
            df_rmsnorm_head<<<dim3(DF_KV, columns), 128, 0, sequence->stream>>>(
                    projected_k, value.knorm, DF_KV);
            df_rope_mrope4<<<dim3(DF_KV, columns), DF_HDIM / 2, 0, sequence->stream>>>(
                    projected_k, DF_KV, device_positions, columns,
                    weights.mrope_sections[0], weights.mrope_sections[1], weights.mrope_sections[2],
                    weights.rope_freq_base);
            df_kv_store_ring<<<dim3((DF_KV * DF_HDIM + 255) / 256, columns), 256, 0, sequence->stream>>>(
                    projected_k, projected_v, scratch.k_cache[layer], scratch.v_cache[layer],
                    device_positions, columns);
        }
        if (!hip_ok(runtime, hipGetLastError(), "launch DFlash feature injection") ||
                !hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize DFlash feature injection")) {
            return false;
        }
        sequence->coverage_end = positions[columns - 1] + 1;
        sequence->coverage_begin = std::max(sequence->coverage_begin,
                sequence->coverage_end - DF_WINDOW);
    }
    sequence->logical_pos = sequence->coverage_end;
    sequence->generation++;
    sequence->has_multimodal_positions = sequence->has_multimodal_positions || multimodal;
    return true;
}

static void enqueue_dflash_graph(spec_sequence * sequence) {
    auto & scratch = sequence->dflash;
    auto & weights = sequence->runtime->dflash;
    df_embed_rows<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, sequence->stream>>>(
            weights.target_embedding, scratch.tokens, scratch.x, DF_EMBD);
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        auto & value = weights.layers[layer];
        df_rmsnorm<<<DF_BLOCK, 256, 0, sequence->stream>>>(scratch.x, value.attn_norm, scratch.xn, DF_EMBD);
        dflash_gemv_q4(DF_BLOCK, value.conv_a_proj, scratch.xn, scratch.dyn_a,
                DF_EMBD, DF_CONVDYN, sequence->stream);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.xn, scratch.dyn_a, value.conv_a_base, scratch.conv, 0, DF_BLOCK);
        dflash_gemv_q4(DF_BLOCK, value.wq, scratch.conv, scratch.q,
                DF_EMBD, DF_HEADS * DF_HDIM, sequence->stream);
        dflash_gemv_q4(DF_BLOCK, value.wk, scratch.conv, scratch.k,
                DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
        if (value.wv_q6) {
            dflash_gemv_q6(DF_BLOCK, static_cast<const block_q6_K *>(value.wv), scratch.conv,
                    scratch.v, DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
        } else {
            dflash_gemv_q4(DF_BLOCK, static_cast<const block_q4_K *>(value.wv), scratch.conv,
                    scratch.v, DF_EMBD, DF_KV * DF_HDIM, sequence->stream);
        }
        df_rmsnorm_head<<<dim3(DF_HEADS, DF_BLOCK), 128, 0, sequence->stream>>>(
                scratch.q, value.qnorm, DF_HEADS);
        df_rmsnorm_head<<<dim3(DF_KV, DF_BLOCK), 128, 0, sequence->stream>>>(
                scratch.k, value.knorm, DF_KV);
        df_rope_mrope4<<<dim3(DF_HEADS, DF_BLOCK), DF_HDIM / 2, 0, sequence->stream>>>(
                scratch.q, DF_HEADS, scratch.positions, DF_BLOCK,
                weights.mrope_sections[0], weights.mrope_sections[1], weights.mrope_sections[2],
                weights.rope_freq_base);
        df_rope_mrope4<<<dim3(DF_KV, DF_BLOCK), DF_HDIM / 2, 0, sequence->stream>>>(
                scratch.k, DF_KV, scratch.positions, DF_BLOCK,
                weights.mrope_sections[0], weights.mrope_sections[1], weights.mrope_sections[2],
                weights.rope_freq_base);
        df_attn_ring<<<dim3(DF_HEADS, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.q, scratch.k_cache[layer], scratch.v_cache[layer], scratch.k, scratch.v,
                scratch.attention, scratch.positions);
        dflash_gemv_q4(DF_BLOCK, value.wo, scratch.attention, scratch.output,
                DF_HEADS * DF_HDIM, DF_EMBD, sequence->stream);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.output, scratch.dyn_a, value.conv_a_base, scratch.conv, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, sequence->stream>>>(
                scratch.x, scratch.conv, DF_BLOCK * DF_EMBD);
        df_rmsnorm<<<DF_BLOCK, 256, 0, sequence->stream>>>(scratch.x, value.ffn_norm, scratch.xn, DF_EMBD);
        dflash_gemv_q4(DF_BLOCK, value.conv_f_proj, scratch.xn, scratch.dyn_f,
                DF_EMBD, DF_CONVDYN, sequence->stream);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.xn, scratch.dyn_f, value.conv_f_base, scratch.conv, 0, DF_BLOCK);
        dflash_gemv_q4(DF_BLOCK, value.up, scratch.conv, scratch.ffn_up,
                DF_EMBD, DF_FF, sequence->stream);
        dflash_gemv_q4(DF_BLOCK, value.gate, scratch.conv, scratch.ffn_gate,
                DF_EMBD, DF_FF, sequence->stream);
        df_silu_mul<<<(DF_BLOCK * DF_FF + 255) / 256, 256, 0, sequence->stream>>>(
                scratch.ffn_gate, scratch.ffn_up, DF_BLOCK * DF_FF);
        if (value.down_q6) {
            dflash_gemv_q6(DF_BLOCK, static_cast<const block_q6_K *>(value.down), scratch.ffn_gate,
                    scratch.ffn_tmp, DF_FF, DF_EMBD, sequence->stream);
        } else {
            dflash_gemv_q4(DF_BLOCK, static_cast<const block_q4_K *>(value.down), scratch.ffn_gate,
                    scratch.ffn_tmp, DF_FF, DF_EMBD, sequence->stream);
        }
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.ffn_tmp, scratch.dyn_f, value.conv_f_base, scratch.conv, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, sequence->stream>>>(
                scratch.x, scratch.conv, DF_BLOCK * DF_EMBD);
    }
    df_rmsnorm<<<DF_BLOCK, 256, 0, sequence->stream>>>(
            scratch.x, weights.output_norm, scratch.token_embedding, DF_EMBD);
    gemv8t_q6_K<DF_BLOCK><<<(40960 + 7) / 8, 256, 0, sequence->stream>>>(
            weights.target_head, scratch.token_embedding, scratch.logits, DF_EMBD, 40960);
    dflash_gemv_q4(DF_BLOCK, weights.selector_hidden, scratch.token_embedding,
            scratch.selector_hidden, DF_EMBD, DF_RANK, sequence->stream);
    for (int slot = 0; slot < DF_TOPK; ++slot) {
        dfa_argmax_partial<<<dim3(scratch.topk_parts, DF_BLOCK), 256, 0, sequence->stream>>>(
                scratch.logits, 40960, scratch.topk_part_max, scratch.topk_part_ids, scratch.topk_parts);
        dfa_argmax_final_mask<<<DF_BLOCK, 1, 0, sequence->stream>>>(
                scratch.logits, 40960, scratch.topk_part_max, scratch.topk_part_ids,
                scratch.topk_parts, scratch.topk_ids, scratch.topk_values, slot, DF_TOPK);
    }
}

static bool ensure_dflash_graph(spec_sequence * sequence) {
    auto & scratch = sequence->dflash;
    if (scratch.graph_exec != nullptr) {
        return true;
    }
    auto * runtime = sequence->runtime;
    hipGraph_t graph = nullptr;
    if (!hip_ok(runtime, hipStreamBeginCapture(sequence->stream, hipStreamCaptureModeThreadLocal),
                "begin DFlash graph capture")) {
        return false;
    }
    enqueue_dflash_graph(sequence);
    // End capture even when a kernel launch reports an error.  Otherwise HIP
    // leaves this per-sequence stream in capture mode and a recoverable
    // DFlash fault can poison later cleanup or other speculative sources.
    const hipError_t launch_status = hipGetLastError();
    const hipError_t end_status = hipStreamEndCapture(sequence->stream, &graph);
    if (!hip_ok(runtime, launch_status, "capture DFlash draft kernels") ||
            !hip_ok(runtime, end_status, "end DFlash graph capture")) {
        if (graph != nullptr) {
            hipGraphDestroy(graph);
        }
        return false;
    }
    const hipError_t instantiate = hipGraphInstantiate(&scratch.graph_exec, graph, nullptr, nullptr, 0);
    hipGraphDestroy(graph);
    return hip_ok(runtime, instantiate, "instantiate DFlash draft graph");
}

static bool launch_dflash_draft(spec_sequence * sequence, ggml_spec_accel_dflash_draft * io) {
    auto * runtime = sequence->runtime;
    auto & scratch = sequence->dflash;
    auto & weights = runtime->dflash;
    if (io->last_token < 0 || io->last_token >= SPEC_VOCAB || io->past_tokens < 1 ||
            io->max_draft == 0 || io->max_draft > DF_BLOCK - 1 || io->output_ids == nullptr ||
            io->output_capacity < io->max_draft ||
            (io->output_confidences != nullptr && io->confidence_capacity < io->max_draft) ||
            uint32_t(io->past_tokens + DF_BLOCK) > runtime->descriptor.n_ctx) {
        runtime->error = "invalid DFlash draft request";
        return false;
    }
    const int32_t required_begin = std::max(0, io->past_tokens - (DF_WINDOW - 1));
    if (sequence->coverage_begin > required_begin || sequence->coverage_end < io->past_tokens) {
        return false;
    }

    std::array<int32_t, DF_BLOCK> tokens {};
    std::array<int32_t, DF_BLOCK * 4> positions {};
    tokens[0] = io->last_token;
    for (int row = 1; row < DF_BLOCK; ++row) {
        tokens[row] = weights.mask_token_id;
    }
    for (int plane = 0; plane < 4; ++plane) {
        for (int row = 0; row < DF_BLOCK; ++row) {
            positions[size_t(plane) * DF_BLOCK + row] = io->past_tokens + row;
        }
    }
    if (!hip_ok(runtime, hipMemcpyAsync(scratch.tokens, tokens.data(), sizeof(tokens),
                    hipMemcpyHostToDevice, sequence->stream), "upload DFlash draft tokens") ||
            !hip_ok(runtime, hipMemcpyAsync(scratch.positions, positions.data(), sizeof(positions),
                    hipMemcpyHostToDevice, sequence->stream), "upload DFlash draft positions") ||
            !ensure_dflash_graph(sequence) ||
            !hip_ok(runtime, hipGraphLaunch(scratch.graph_exec, sequence->stream), "launch DFlash draft graph")) {
        return false;
    }

    std::array<int32_t, DF_BLOCK * DF_TOPK> candidate_ids {};
    std::array<float, DF_BLOCK * DF_TOPK> candidate_values {};
    std::array<float, DF_BLOCK * DF_RANK> selector_hidden {};
    if (!hip_ok(runtime, hipMemcpyAsync(candidate_ids.data(), scratch.topk_ids, sizeof(candidate_ids),
                    hipMemcpyDeviceToHost, sequence->stream), "download DFlash candidate IDs") ||
            !hip_ok(runtime, hipMemcpyAsync(candidate_values.data(), scratch.topk_values, sizeof(candidate_values),
                    hipMemcpyDeviceToHost, sequence->stream), "download DFlash candidate values") ||
            !hip_ok(runtime, hipMemcpyAsync(selector_hidden.data(), scratch.selector_hidden, sizeof(selector_hidden),
                    hipMemcpyDeviceToHost, sequence->stream), "download DFlash selector state") ||
            !hip_ok(runtime, hipStreamSynchronize(sequence->stream), "synchronize DFlash draft")) {
        return false;
    }

    int predecessor_index = 0;
    uint32_t output_count = 0;
    for (int row = 1; row < DF_BLOCK && output_count < io->max_draft; ++row) {
        const float * hidden = selector_hidden.data() + size_t(row) * DF_RANK;
        const int predecessor_slice = row == 1 ? -1 :
                candidate_ids[size_t(row - 1) * DF_TOPK + predecessor_index];
        const int32_t predecessor = row == 1 ? io->last_token :
                (predecessor_slice >= 0 && predecessor_slice < 40960 ?
                    weights.remap[predecessor_slice] : -1);
        if (predecessor < 0 || predecessor >= SPEC_VOCAB) {
            break;
        }
        const float * previous = weights.selector_prev.data() + size_t(predecessor) * DF_RANK;
        float best = -std::numeric_limits<float>::infinity();
        int best_index = -1;
        std::array<float, DF_TOPK> selector_scores {};
        selector_scores.fill(-std::numeric_limits<float>::infinity());
        for (int candidate = 0; candidate < DF_TOPK; ++candidate) {
            const int slice = candidate_ids[size_t(row) * DF_TOPK + candidate];
            if (slice < 0 || slice >= 40960) {
                continue;
            }
            const int32_t token = weights.remap[slice];
            const float * next = weights.selector_next.data() + size_t(token) * DF_RANK;
            float score = candidate_values[size_t(row) * DF_TOPK + candidate];
            for (int rank = 0; rank < DF_RANK; ++rank) {
                score += next[rank] * previous[rank] * hidden[rank];
            }
            selector_scores[candidate] = score;
            if (std::isfinite(score) && (best_index < 0 || score > best)) {
                best = score;
                best_index = candidate;
            }
        }
        if (best_index < 0) {
            break;
        }
        predecessor_index = best_index;
        const int slice = candidate_ids[size_t(row) * DF_TOPK + best_index];
        if (slice < 0 || slice >= 40960) {
            break;
        }
        io->output_ids[output_count++] = weights.remap[slice];
        if (io->output_confidences != nullptr) {
            float normalizer = 0.0f;
            for (float score : selector_scores) {
                if (std::isfinite(score)) {
                    normalizer += std::exp(score - best);
                }
            }
            io->output_confidences[output_count - 1] = normalizer > 0.0f ? 1.0f / normalizer : 0.0f;
        }
    }
    io->output_count = output_count;
    return true;
}
