#pragma once

#include <algorithm>
#include <cstdint>

struct ggml_vk_fattn_kvarn_plan_input {
    uint32_t n_kv;
    uint32_t n_query;
    uint32_t n_query_heads;
    uint32_t n_kv_heads;
    uint32_t n_stream;
    uint32_t shader_core_count;
    uint32_t tail_tokens = 0;
    uint32_t head_dim = 0;
    uint64_t workspace_budget_bytes = 128ull * 1024ull * 1024ull;
    uint32_t gqa_group_limit = 4;
    bool force_decode_body_tiles = false;
    uint32_t queries_per_workgroup = 1;
    // Non-zero when a caller has already selected the body split geometry for
    // a backend-native kernel.  The workspace planner may still tile queries,
    // but must preserve the supplied body partial layout.
    uint32_t body_splits_override = 0;
};

struct ggml_vk_fattn_kvarn_plan_result {
    uint32_t gqa_group_size;
    uint32_t workgroups_y;
    uint32_t split_k;
    uint32_t body_splits;
    uint32_t tail_splits;
    uint32_t max_tail_tokens_per_split;
    uint32_t query_tile;
    uint32_t query_tiles;
    uint64_t workspace_bytes;
    bool workspace_supported;
};

inline ggml_vk_fattn_kvarn_plan_result ggml_vk_fattn_kvarn_plan(
        const ggml_vk_fattn_kvarn_plan_input & input) {
    const uint32_t gqa = input.n_query_heads / input.n_kv_heads;
    const uint32_t gqa_group_size = std::min(
        gqa, std::max(1u, input.gqa_group_limit));
    const uint32_t gqa_groups = (gqa + gqa_group_size - 1) / gqa_group_size;
    const uint32_t workgroups_y = input.n_kv_heads * gqa_groups;
    const uint32_t packed_query_groups = input.n_query == 0 ? 0 :
        (input.n_query + std::max(1u, input.queries_per_workgroup) - 1u) /
            std::max(1u, input.queries_per_workgroup);
    const uint32_t workgroups = std::max(
        1u, packed_query_groups * workgroups_y * input.n_stream);
    const uint32_t target_workgroups = std::max(1u, input.shader_core_count) * 2u;

    // Decode and short speculative verification are latency-bound. Give every
    // 64-token half-record its own partial so context growth creates more work
    // groups instead of a longer serial loop inside a fixed dispatch.
    uint32_t body_splits = 0;
    if (input.n_kv > 0) {
        if (input.body_splits_override > 0) {
            body_splits = input.body_splits_override;
        } else if (input.n_query <= 8 || input.force_decode_body_tiles) {
            body_splits = (input.n_kv + 63u) / 64u;
        } else {
            body_splits = std::max(1u,
                (target_workgroups + workgroups - 1u) / workgroups);
            body_splits = std::min(body_splits, (input.n_kv + 127u) / 128u);
        }
    }

    uint32_t tail_splits = input.tail_tokens > 0 ?
        (input.tail_tokens + 63u) / 64u : 0u;
    tail_splits = std::min(tail_splits, 255u);
    body_splits = std::min(body_splits, 65535u);

    uint32_t split_k = body_splits + tail_splits;
    if (split_k == 0) {
        split_k = 1;
    }

    // Keep body half-records and exact-tail 64-token chunks independent.  If
    // reducing every query at once would exceed the transient budget, tile the
    // query dimension instead of coalescing token chunks and reintroducing a
    // serial tail.  Reducing the body split count is only a last-resort route
    // for shapes where even a single query would not fit.
    const uint64_t bytes_per_query_split = input.head_dim == 0 ? 0 :
        (uint64_t(input.head_dim) + 2u) * sizeof(float) *
        input.n_query_heads * input.n_stream;
    bool workspace_supported = input.tail_tokens <= 255u * 64u;
    if (bytes_per_query_split > 0 && split_k > 0 &&
            bytes_per_query_split * split_k > input.workspace_budget_bytes) {
        const uint32_t budget_splits = uint32_t(
            input.workspace_budget_bytes / bytes_per_query_split);
        if (budget_splits < tail_splits || budget_splits == 0) {
            workspace_supported = false;
        } else if (input.body_splits_override == 0) {
            body_splits = std::min(body_splits, budget_splits - tail_splits);
            split_k = body_splits + tail_splits;
            if (split_k == 0) {
                split_k = 1;
            }
        } else {
            workspace_supported = false;
        }
    }

    uint32_t query_tile = std::max(1u, input.n_query);
    if (bytes_per_query_split > 0 && workspace_supported) {
        const uint64_t bytes_per_query = bytes_per_query_split * split_k;
        const uint64_t max_queries = bytes_per_query == 0 ? UINT32_MAX :
            input.workspace_budget_bytes / bytes_per_query;
        if (max_queries == 0) {
            workspace_supported = false;
            query_tile = 1;
        } else {
            query_tile = std::min<uint32_t>(query_tile,
                uint32_t(std::min<uint64_t>(max_queries, UINT32_MAX)));
        }
    }
    const uint32_t query_tiles = input.n_query == 0 ? 0 :
        (input.n_query + query_tile - 1u) / query_tile;
    const uint64_t workspace_bytes = bytes_per_query_split * split_k * query_tile;
    const uint32_t max_tail_tokens_per_split = tail_splits > 0 ?
        std::min(input.tail_tokens, 64u) : 0u;
    return { gqa_group_size, workgroups_y, split_k, body_splits,
        tail_splits, max_tail_tokens_per_split, query_tile, query_tiles,
        workspace_bytes, workspace_supported };
}
