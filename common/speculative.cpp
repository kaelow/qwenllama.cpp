#include "speculative.h"

#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-spec-accel.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <stdexcept>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    virtual bool adaptive_dm_supported() const { return false; }

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual bool validate_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & data) const {
        return data.empty();
    }
    virtual bool set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & data) {
        return data.empty();
    }

    // Persistent prompt caches need self-contained speculative state.  The
    // default is suitable for host-only implementations; device accelerators
    // override these without inflating ordinary rollback checkpoints.
    virtual bool get_state_durable(llama_seq_id seq_id, std::vector<uint8_t> & data) const {
        return get_state(seq_id, data);
    }
    virtual bool validate_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) const {
        return validate_state(seq_id, data);
    }
    virtual bool set_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) {
        return set_state(seq_id, data);
    }
    virtual void cancel_state_durable(llama_seq_id /*seq_id*/) {}

    // Sequence lifecycle mirrors llama_memory operations.  Host/native
    // implementations can reconstruct their small bookkeeping through the
    // compact state path; device sidecars override these to keep cache moves
    // on the accelerator.
    virtual bool sequence_copy(llama_seq_id source, llama_seq_id destination) {
        std::vector<uint8_t> state;
        return !get_state(source, state) ||
                (validate_state(destination, state) && set_state(destination, state));
    }
    virtual bool sequence_remove_suffix(llama_seq_id /*seq_id*/, int32_t /*pos*/) {
        return true;
    }
    virtual bool sequence_shift(
            llama_seq_id /*seq_id*/, int32_t /*begin*/, int32_t /*end*/, int32_t /*delta*/) {
        return true;
    }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos_next, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos_next + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return data.empty();
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            return true;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return false;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));
        return pos >= 0;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!validate_state(seq_id, data)) {
            return false;
        }
        if (!need_boundary_stash()) {
            return true;
        }
        if (data.empty()) {
            pending_pos_last[seq_id] = -1;
            std::fill(pending_g_last[seq_id].begin(), pending_g_last[seq_id].end(), 0.0f);
            return true;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        GGML_ASSERT(pending_g_last[seq_id].size() == (size_t) n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
        return true;
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        if (!common_speculative_dflash_adaptive_dm_supported(selector_top_k)) {
            LOG_INF("%s: DFlash2 uses its fixed block limit and selector confidence; Bee adaptive draft-max is disabled\n", __func__);
        }
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // Extract all target taps as one token-major output. This turns five
        // backend transfers plus a nested host gather into one transfer.
        std::vector<uint32_t> target_layer_bundle(target_layer_ids_n);
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            target_layer_bundle[k] = (uint32_t) target_layer_ids[k];
        }
        llama_set_embeddings_layer_inp_bundle(
                ctx_tgt, target_layer_bundle.data(), target_layer_bundle.size());

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);
        const float * target_features = llama_get_embeddings_layer_inp_bundle(ctx_tgt);
        GGML_ASSERT(target_features && "DFlash bundled target features were not extracted");

        auto source_pos = [&](int plane, int32_t idx) -> llama_pos {
            if (is_mrope && has_embeddings) {
                return batch_in.pos[(size_t) plane * n_tokens + idx];
            }
            const llama_pos p = batch_in.pos[idx];
            return plane == 3 ? 0 : p;
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                float * feature_chunk = const_cast<float *>(target_features +
                        (size_t) (i_batch_beg[seq_id] + offset) * n_embd_enc);

                // fuse extracted features through DFlash encoder
                // M-RoPE drafts read 4 position rows per token from embd batches, so pass them explicitly
                std::vector<llama_pos> enc_pos;
                if (is_mrope) {
                    enc_pos.resize((size_t) 4 * n_chunk);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const int32_t idx = i_batch_beg[seq_id] + offset + i;
                        for (int plane = 0; plane < 4; ++plane) {
                            enc_pos[(size_t) plane * n_chunk + i] = source_pos(plane, idx);
                        }
                    }
                }

                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ feature_chunk,
                    /*.pos      =*/ is_mrope ? enc_pos.data() : nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const int32_t idx = i_batch_beg[seq_id] + offset + i;
                    batch_inject.pos[i] = source_pos(0, idx);
                    if (is_mrope) {
                        for (int plane = 1; plane < 4; ++plane) {
                            batch_inject.pos[(size_t) plane * n_chunk + i] = source_pos(plane, idx);
                        }
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos_next;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + selector_top_k));
                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool adaptive_dm_supported() const override {
        return common_speculative_dflash_adaptive_dm_supported(selector_top_k);
    }
};

// Backend-owned Qwen3.8 DFlash2 accelerator.  The target context remains a
// normal llama_context with whichever standard or KVarN cache pair the user
// selected; this implementation only consumes the bundled target-layer output
// and owns its independent F16 sliding-window draft state in ggml-hip.
struct common_speculative_impl_draft_dflash_hip : public common_speculative_impl {
    common_params_speculative_draft params;

    const ggml_spec_accel_api * api = nullptr;
    ggml_spec_accel_runtime_t runtime = nullptr;
    std::vector<ggml_spec_accel_seq_t> sequences;

    int32_t feature_width = 0;
    std::vector<int32_t> verify_pos_begin;
    std::vector<int32_t> verify_rows;
    std::vector<bool> disabled;

    struct checkpoint_state {
        uint32_t magic;
        uint32_t version;
        int32_t logical_pos;
        int32_t coverage_begin;
        int32_t coverage_end;
        uint32_t reserved;
        uint64_t generation;
    };

    static constexpr uint32_t state_magic = 0x48464442; // BDFH
    static constexpr uint32_t state_version = 1;

    static int32_t device_ordinal(const char * name) {
        if (name == nullptr) {
            return 0;
        }
        const std::string value(name);
        size_t begin = value.size();
        while (begin > 0 && std::isdigit(static_cast<unsigned char>(value[begin - 1]))) {
            --begin;
        }
        if (begin == value.size()) {
            return 0;
        }
        try {
            return std::stoi(value.substr(begin));
        } catch (...) {
            return 0;
        }
    }

    [[noreturn]] void startup_error(const std::string & message) const {
        throw std::runtime_error("HIP DFlash accelerator: " + message);
    }

    std::string backend_error() const {
        if (api != nullptr && api->last_error != nullptr) {
            const char * message = api->last_error(runtime);
            if (message != nullptr && message[0] != '\0') {
                return message;
            }
        }
        return "backend operation failed";
    }

    void disable_sequence(llama_seq_id seq_id, const char * operation, int32_t status) {
        if (!disabled[seq_id]) {
            SPC_WRN("HIP DFlash disabled for seq_id=%d after %s failed (status=%d): %s; "
                    "other speculative sources and normal generation remain available\n",
                    (int) seq_id, operation, status, backend_error().c_str());
        }
        disabled[seq_id] = true;
    }

    void release_backend() noexcept {
        if (api == nullptr) {
            return;
        }
        for (auto & sequence : sequences) {
            if (sequence != nullptr) {
                api->sequence_free(sequence);
                sequence = nullptr;
            }
        }
        if (runtime != nullptr) {
            api->runtime_free(runtime);
            runtime = nullptr;
        }
    }

    common_speculative_impl_draft_dflash_hip(
            const common_params_speculative & all_params,
            uint32_t n_seq)
    try : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, n_seq,
                std::min(all_params.draft.n_max, 7))
        , params(all_params.draft) {
        if (params.ctx_tgt == nullptr) {
            startup_error("target context is unavailable");
        }
        if (params.accelerator_model.empty()) {
            startup_error("--spec-draft-accelerator-model is required when hip is selected");
        }
        if (params.target_model_path.empty()) {
            startup_error("target model path is unavailable for artifact identity validation");
        }

        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        llama_spec_accel_model_desc model_desc {};
        if (!llama_model_get_spec_accel_desc(model_tgt, &model_desc) || !model_desc.is_mrope) {
            startup_error("target is not a supported Qwen3.8 M-RoPE model");
        }

        ggml_backend_dev_t selected = nullptr;
        auto try_device = [&](ggml_backend_dev_t device) {
            if (device == nullptr) {
                return false;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
            if (reg == nullptr) {
                return false;
            }
            auto getter = reinterpret_cast<ggml_backend_spec_accel_get_api_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_spec_accel_get_api"));
            if (getter == nullptr) {
                return false;
            }
            const ggml_spec_accel_api * candidate = getter(GGML_SPEC_ACCEL_ABI_VERSION);
            if (candidate == nullptr || candidate->abi_version != GGML_SPEC_ACCEL_ABI_VERSION ||
                    candidate->struct_size < sizeof(ggml_spec_accel_api)) {
                return false;
            }
            selected = device;
            api = candidate;
            return true;
        };
        for (ggml_backend_dev_t device : params.devices) {
            if (try_device(device)) {
                break;
            }
        }
        if (selected == nullptr && params.devices.empty()) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                if (try_device(ggml_backend_dev_get(i))) {
                    break;
                }
            }
        }
        if (selected == nullptr || api == nullptr) {
            startup_error("the selected draft device does not expose the accelerator ABI; "
                    "use a build with GGML_HIP_SPEC_ACCEL=ON and select the gfx1100 ROCm entry "
                    "reported by --list-devices (for example ROCm1)");
        }

        const char * selected_name = ggml_backend_dev_name(selected);
        ggml_spec_accel_descriptor descriptor {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = GGML_SPEC_ACCEL_KIND_DFLASH;
        descriptor.artifact_path = params.accelerator_model.c_str();
        descriptor.device_name = selected_name;
        descriptor.device_ordinal = device_ordinal(selected_name);
        descriptor.n_embd = model_desc.n_embd;
        descriptor.n_vocab = model_desc.n_vocab;
        descriptor.n_head = model_desc.n_head;
        descriptor.n_head_kv = model_desc.n_head_kv;
        descriptor.head_dim = model_desc.head_dim;
        descriptor.n_rot = model_desc.n_rot;
        descriptor.n_ctx = llama_n_ctx(params.ctx_tgt);
        descriptor.n_seq_max = n_seq;
        // This is DFlash's private bounded cache, not the target model's cache.
        descriptor.cache_type = GGML_SPEC_ACCEL_CACHE_F16;
        descriptor.target_model_path = params.target_model_path.c_str();

        ggml_spec_accel_capabilities capabilities {};
        capabilities.struct_size = sizeof(capabilities);
        int32_t status = api->query_capabilities(&descriptor, &capabilities);
        if (status != GGML_SPEC_ACCEL_STATUS_OK ||
                !(capabilities.flags & GGML_SPEC_ACCEL_CAP_DFLASH) ||
                capabilities.target_layer_count == 0 ||
                capabilities.target_layer_count > 8 || capabilities.feature_width == 0) {
            startup_error(string_format(
                    "device/model/artifact combination is unsupported (status=%d)", status));
        }

        std::vector<uint32_t> target_layers(capabilities.target_layer_count);
        const uint32_t target_layer_count = llama_model_n_layer(model_tgt);
        for (uint32_t i = 0; i < capabilities.target_layer_count; ++i) {
            if (capabilities.target_layers[i] < 0 ||
                    uint32_t(capabilities.target_layers[i]) > target_layer_count) {
                startup_error("artifact requests a target feature layer outside the model");
            }
            target_layers[i] = uint32_t(capabilities.target_layers[i]);
        }
        feature_width = int32_t(capabilities.feature_width);
        if (feature_width != int32_t(target_layers.size()) * llama_model_n_embd(model_tgt)) {
            startup_error("artifact feature width does not match its target-layer bundle");
        }

        status = api->runtime_create(&descriptor, &runtime);
        if (status != GGML_SPEC_ACCEL_STATUS_OK || runtime == nullptr) {
            startup_error(string_format("runtime creation failed (status=%d): %s",
                    status, backend_error().c_str()));
        }
        sequences.assign(n_seq, nullptr);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            status = api->sequence_create(runtime, seq_id, &sequences[seq_id]);
            if (status != GGML_SPEC_ACCEL_STATUS_OK || sequences[seq_id] == nullptr) {
                startup_error(string_format("sequence %d creation failed (status=%d): %s",
                        (int) seq_id, status, backend_error().c_str()));
            }
        }

        llama_set_embeddings_layer_inp_bundle(
                params.ctx_tgt, target_layers.data(), target_layers.size());
        verify_pos_begin.assign(n_seq, -1);
        verify_rows.assign(n_seq, 0);
        disabled.assign(n_seq, false);
        this->params.n_max = std::min(this->params.n_max, 7);
        this->params.n_min = std::min(this->params.n_min, 7);
        this->n_max = this->params.n_max;

        SPC_INF("HIP DFlash2 accelerator enabled on %s: private-cache=f16/window-%u, "
                "target-kv=independent/unrestricted, slots=%u, artifact='%s'\n",
                selected_name ? selected_name : "ROCm", capabilities.cache_window,
                n_seq, params.accelerator_model.c_str());
    } catch (...) {
        release_backend();
        throw;
    }

    ~common_speculative_impl_draft_dflash_hip() override {
        release_backend();
    }

    void begin(llama_seq_id, const llama_tokens &) override {
        // Resident coverage is selected/restored by the checkpoint path.
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }
        const bool has_tokens = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd != nullptr;
        if (has_tokens == has_embeddings || batch_in.pos == nullptr) {
            return true;
        }
        const float * bundle = llama_get_embeddings_layer_inp_bundle(params.ctx_tgt);
        if (bundle == nullptr) {
            SPC_WRN("%s", "target feature bundle is unavailable; HIP DFlash skipped for this batch\n");
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;
        std::vector<std::vector<int32_t>> indices(n_seq);
        for (int32_t row = 0; row < n_tokens; ++row) {
            if (batch_in.n_seq_id[row] != 1 || batch_in.seq_id[row] == nullptr) {
                continue;
            }
            const llama_seq_id seq_id = batch_in.seq_id[row][0];
            if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
                indices[seq_id].push_back(row);
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const auto & rows = indices[seq_id];
            if (rows.empty()) {
                continue;
            }
            const uint32_t count = uint32_t(rows.size());
            verify_pos_begin[seq_id] = batch_in.pos[rows.front()];
            verify_rows[seq_id] = int32_t(count);
            if (disabled[seq_id]) {
                continue;
            }

            std::vector<float> features(size_t(count) * feature_width);
            std::vector<int32_t> positions(size_t(count) * 4);
            for (uint32_t row = 0; row < count; ++row) {
                const int32_t source = rows[row];
                std::memcpy(features.data() + size_t(row) * feature_width,
                        bundle + size_t(source) * feature_width,
                        size_t(feature_width) * sizeof(float));
                for (uint32_t plane = 0; plane < 4; ++plane) {
                    positions[size_t(plane) * count + row] = has_tokens ?
                            batch_in.pos[source] :
                            batch_in.pos[size_t(plane) * n_tokens + source];
                }
            }
            ggml_spec_accel_dflash_features input {};
            input.struct_size = sizeof(input);
            input.count = count;
            input.features = features.data();
            input.feature_width = feature_width;
            input.positions = positions.data();
            input.position_planes = 4;
            const int32_t status = api->dflash_features(sequences[seq_id], &input);
            if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                disable_sequence(seq_id, "feature injection", status);
            }
        }
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        std::array<llama_token, 7> output {};
        std::array<float, 7> confidence {};
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting || disabled[seq_id]) {
                continue;
            }
            int32_t limit = params.n_max;
            if (dp.n_max >= 0) {
                limit = std::min(limit, dp.n_max);
            }
            limit = std::min(limit, 7);
            if (limit <= 0 || limit < params.n_min) {
                continue;
            }

            ggml_spec_accel_dflash_draft request {};
            request.struct_size = sizeof(request);
            request.last_token = dp.id_last;
            request.past_tokens = dp.pos_next;
            request.max_draft = limit;
            request.output_ids = output.data();
            request.output_capacity = output.size();
            request.output_confidences = confidence.data();
            request.confidence_capacity = confidence.size();
            const int32_t status = api->dflash_draft(sequences[seq_id], &request);
            if (status == GGML_SPEC_ACCEL_STATUS_NO_COVERAGE ||
                    status == GGML_SPEC_ACCEL_STATUS_UNAVAILABLE) {
                continue;
            }
            if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                disable_sequence(seq_id, "draft", status);
                continue;
            }
            auto & result = *dp.result;
            for (uint32_t i = 0; i < request.output_count; ++i) {
                if (params.p_min > 0.0f && confidence[i] < params.p_min) {
                    break;
                }
                result.push_back(output[i]);
            }
            if (result.size() < size_t(params.n_min)) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || disabled[seq_id] ||
                verify_pos_begin[seq_id] < 0 || verify_rows[seq_id] <= 0) {
            return;
        }
        const int32_t accepted_row = std::min<int32_t>(n_accepted, verify_rows[seq_id] - 1);
        const int32_t keep_end = verify_pos_begin[seq_id] + accepted_row + 1;
        const int32_t status = api->sequence_remove_suffix(sequences[seq_id], keep_end);
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            disable_sequence(seq_id, "accepted-suffix truncation", status);
        }
    }

    bool decode_state(
            llama_seq_id seq_id,
            const std::vector<uint8_t> & data,
            checkpoint_state & state) const {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || data.size() != sizeof(state)) {
            return false;
        }
        std::memcpy(&state, data.data(), sizeof(state));
        return state.magic == state_magic && state.version == state_version && state.reserved == 0 &&
                state.logical_pos >= 0 && state.coverage_begin >= 0 &&
                state.coverage_end >= state.coverage_begin && state.logical_pos <= state.coverage_end;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        if (api->sequence_plan(sequences[seq_id], &plan) != GGML_SPEC_ACCEL_STATUS_OK) {
            return false;
        }
        checkpoint_state state {
            state_magic, state_version, plan.logical_pos,
            plan.coverage_begin, plan.coverage_end, 0, plan.generation,
        };
        data.resize(sizeof(state));
        std::memcpy(data.data(), &state, sizeof(state));
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (data.empty()) {
            return seq_id >= 0 && seq_id < (llama_seq_id) n_seq;
        }
        checkpoint_state state {};
        // Coverage is deliberately not consulted here.  A syntactically valid
        // old checkpoint remains restorable even after its DFlash ring window
        // has been overwritten; speculation then waits for a fresh window.
        return decode_state(seq_id, data, state);
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        verify_pos_begin[seq_id] = -1;
        verify_rows[seq_id] = 0;
        if (data.empty()) {
            const int32_t status = api->sequence_reset(sequences[seq_id]);
            disabled[seq_id] = status != GGML_SPEC_ACCEL_STATUS_OK;
            return status == GGML_SPEC_ACCEL_STATUS_OK;
        }
        checkpoint_state state {};
        if (!decode_state(seq_id, data, state)) {
            return false;
        }
        ggml_spec_accel_checkpoint_desc descriptor {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.logical_pos = state.logical_pos;
        descriptor.coverage_begin = state.coverage_begin;
        descriptor.coverage_end = state.coverage_end;
        descriptor.generation = state.generation;
        ggml_spec_accel_checkpoint_t transaction = nullptr;
        int32_t status = api->checkpoint_validate(sequences[seq_id], &descriptor, &transaction);
        if (status == GGML_SPEC_ACCEL_STATUS_OK && transaction != nullptr) {
            status = api->checkpoint_commit(transaction);
            api->checkpoint_free(transaction);
            disabled[seq_id] = status != GGML_SPEC_ACCEL_STATUS_OK;
            return status == GGML_SPEC_ACCEL_STATUS_OK;
        }
        if (transaction != nullptr) {
            api->checkpoint_free(transaction);
        }
        if (status == GGML_SPEC_ACCEL_STATUS_NO_COVERAGE) {
            // Correctness-preserving fallback: target checkpoint restoration
            // succeeds; only DFlash is unavailable until 2048 new target rows
            // refill its independent sliding window.
            status = api->sequence_invalidate(sequences[seq_id], state.logical_pos);
            disabled[seq_id] = status != GGML_SPEC_ACCEL_STATUS_OK;
            return status == GGML_SPEC_ACCEL_STATUS_OK;
        }
        return false;
    }

    bool get_state_durable(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        data.clear();
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_blob blob {};
        blob.struct_size = sizeof(blob);
        const int32_t status = api->state_export(sequences[seq_id], &blob);
        if (status != GGML_SPEC_ACCEL_STATUS_OK || blob.data == nullptr || blob.size == 0) {
            if (blob.data != nullptr) {
                api->blob_free(&blob);
            }
            return false;
        }
        try {
            const auto * begin = static_cast<const uint8_t *>(blob.data);
            data.assign(begin, begin + blob.size);
        } catch (...) {
            api->blob_free(&blob);
            throw;
        }
        api->blob_free(&blob);
        return true;
    }

    bool validate_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (data.empty()) {
            return seq_id >= 0 && seq_id < (llama_seq_id) n_seq;
        }
        return seq_id >= 0 && seq_id < (llama_seq_id) n_seq &&
                api->state_validate(sequences[seq_id], data.data(), data.size()) ==
                    GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool set_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (data.empty()) {
            return set_state(seq_id, data);
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq ||
                api->state_import(sequences[seq_id], data.data(), data.size()) !=
                    GGML_SPEC_ACCEL_STATUS_OK) {
            return false;
        }
        verify_pos_begin[seq_id] = -1;
        verify_rows[seq_id] = 0;
        disabled[seq_id] = false;
        return true;
    }

    void cancel_state_durable(llama_seq_id seq_id) override {
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
            api->state_cancel(sequences[seq_id]);
        }
    }

    bool sequence_copy(llama_seq_id source, llama_seq_id destination) override {
        if (source < 0 || destination < 0 || source >= (llama_seq_id) n_seq ||
                destination >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        int32_t status = api->sequence_plan(sequences[source], &plan);
        if (status == GGML_SPEC_ACCEL_STATUS_OK && plan.coverage_begin == plan.coverage_end) {
            status = api->sequence_invalidate(sequences[destination], plan.logical_pos);
        } else if (status == GGML_SPEC_ACCEL_STATUS_OK) {
            status = api->sequence_copy(sequences[source], sequences[destination],
                    plan.coverage_begin, plan.coverage_end);
        }
        verify_pos_begin[destination] = -1;
        verify_rows[destination] = 0;
        disabled[destination] = disabled[source] || status != GGML_SPEC_ACCEL_STATUS_OK;
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool sequence_remove_suffix(llama_seq_id seq_id, int32_t pos) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pos < 0) {
            return false;
        }
        const int32_t status = api->sequence_remove_suffix(sequences[seq_id], pos);
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            disable_sequence(seq_id, "sequence suffix removal", status);
        }
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool sequence_shift(llama_seq_id seq_id, int32_t begin, int32_t end, int32_t delta) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || begin < 0 || end < begin || delta == 0) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        int32_t status = api->sequence_plan(sequences[seq_id], &plan);
        const int32_t resident_begin = std::max(begin, plan.coverage_begin);
        const int32_t resident_end = std::min(end, plan.coverage_end);
        if (status == GGML_SPEC_ACCEL_STATUS_OK && resident_begin < resident_end) {
            status = api->sequence_shift(
                    sequences[seq_id], resident_begin, resident_end, delta);
        } else if (status == GGML_SPEC_ACCEL_STATUS_OK) {
            const int64_t shifted = int64_t(plan.logical_pos) + delta;
            status = shifted >= 0 && shifted <= std::numeric_limits<int32_t>::max() ?
                    api->sequence_invalidate(sequences[seq_id], int32_t(shifted)) :
                    GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            disable_sequence(seq_id, "sequence position shift", status);
        }
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    llama_token * batch_token_storage  = nullptr;
    float       * batch_embd_storage   = nullptr;
    float       * batch_hidden_storage = nullptr;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd     = 0;
    int32_t n_embd_inp = 0;

    bool separate_mtp_hidden = false;
    bool is_mrope            = false;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    // Exact source rows per sequence for the current target batch. Reused
    // across process() calls so continuous decode does not allocate each step.
    std::vector<std::vector<int32_t>> batch_rows;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        n_embd     = llama_model_n_embd_out(model_dft);
        n_embd_inp = llama_model_n_embd(model_dft);
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(model_dft));
        separate_mtp_hidden = llama_model_supports_mtp_kv_only(model_dft);
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ 0, /*n_seq_max=*/ 1);
        batch.embd = (float *) malloc(sizeof(float) * (size_t) n_b * std::max(n_embd, n_embd_inp));
        batch.embd_nextn = (float *) malloc(sizeof(float) * (size_t) n_b * n_embd);
        GGML_ASSERT(batch.embd && batch.embd_nextn);

        batch_token_storage  = batch.token;
        batch_embd_storage   = batch.embd;
        batch_hidden_storage = batch.embd_nextn;

        if (is_mrope) {
            free(batch.pos);
            batch.pos = (llama_pos *) malloc(sizeof(llama_pos) * (size_t) 4 * n_b);
            GGML_ASSERT(batch.pos);
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        batch_rows.resize(n_seq);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        // process() switches these pointers to express token versus embedding
        // input. Restore the owning allocations before freeing the batch.
        batch.token      = batch_token_storage;
        batch.embd       = batch_embd_storage;
        batch.embd_nextn = batch_hidden_storage;
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }
        // Legacy MTP graphs use embd for their hidden input and therefore
        // cannot also consume token embeddings. Qwen's separated input path
        // supports both text and multimodal batches.
        if (has_embeddings && !separate_mtp_hidden) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // Keep the exact source-row order for each sequence. Server batches are
        // usually grouped by sequence, but continuous batching is allowed to
        // interleave them. MTP pairs each token with the preceding target hidden
        // row from the same sequence, so a first/last range is not sufficient.
        for (auto & rows : batch_rows) {
            rows.clear();
        }

        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);

            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }

            batch_rows[seq_id].push_back(k);
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            batch.token      = batch_token_storage;
            batch.embd       = separate_mtp_hidden ? nullptr : batch_embd_storage;
            batch.embd_nextn = separate_mtp_hidden ? batch_hidden_storage : nullptr;
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, has_tokens ? batch_in.token[k] : LLAMA_TOKEN_NULL,
                        batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // Text input carries one scalar position per token; the target
            // batch allocator broadcasts it to every M-RoPE plane.  Image
            // embeddings carry four explicit planes.  Preserve that contract
            // here instead of reading beyond a text batch's position array.
            if (is_mrope) {
                for (int plane = 0; plane < 4; ++plane) {
                    for (int k = 0; k < n_tokens; ++k) {
                        batch.pos[(size_t) plane * n_tokens + k] =
                                has_tokens ? batch_in.pos[k] :
                                    batch_in.pos[(size_t) plane * n_tokens + k];
                    }
                }
            }

            if (has_embeddings) {
                GGML_ASSERT(n_embd_inp == llama_model_n_embd(llama_get_model(ctx_tgt)));
                std::memcpy(batch_embd_storage, batch_in.embd,
                        (size_t) n_tokens * n_embd_inp * sizeof(float));

                batch.token = nullptr;
                batch.embd  = batch_embd_storage;
            }

            float * hidden = separate_mtp_hidden ? batch_hidden_storage : batch_embd_storage;

            // Shift target hidden rows within each sequence. The first token of
            // every sequence consumes its cross-batch pending hidden row; every
            // later token consumes the previous source row from that same
            // sequence, regardless of how batch rows are interleaved.
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                const auto & rows = batch_rows[seq_id];
                if (rows.empty()) {
                    continue;
                }

                std::memcpy(hidden + (size_t) rows[0] * n_embd,
                        pending_h[seq_id].data(), row_bytes);

                for (size_t i = 1; i < rows.size(); ++i) {
                    std::memcpy(hidden + (size_t) rows[i] * n_embd,
                            h_tgt + (size_t) rows[i - 1] * n_embd, row_bytes);
                }
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        const auto & rows = batch_rows[seq_id];
                        if (rows.empty()) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[rows.front()], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const auto & rows = batch_rows[seq_id];
            if (rows.empty()) {
                continue;
            }

            const int32_t n_rows = (int32_t) rows.size();
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, rows[i]);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.token      = batch_token_storage;
        batch.embd       = separate_mtp_hidden ? nullptr : batch_embd_storage;
        batch.embd_nextn = separate_mtp_hidden ? batch_hidden_storage : nullptr;
        common_batch_clear(batch);

        float * hidden = separate_mtp_hidden ? batch_hidden_storage : batch_embd_storage;

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.pos_next, { seq_id }, true);
            std::memcpy(hidden + (size_t) (batch.n_tokens - 1) * n_embd,
                    pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos_next, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.pos_next + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(hidden + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.pos_next, { seq_id }, true);
                    std::memcpy(hidden + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.pos_next + i + 1, { seq_id }, true);
                    std::memcpy(hidden + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq ||
                pending_h[seq_id].size() != (size_t) n_embd) {
            return false;
        }

        constexpr uint32_t magic   = 0x3150544d; // MTP1
        constexpr uint32_t version = 1;
        const uint32_t width = uint32_t(n_embd);
        const size_t header_size = sizeof(magic) + sizeof(version) + sizeof(width);
        const size_t row_size = (size_t) n_embd * sizeof(float);

        data.resize(header_size + row_size);
        uint8_t * dst = data.data();
        std::memcpy(dst, &magic, sizeof(magic));
        dst += sizeof(magic);
        std::memcpy(dst, &version, sizeof(version));
        dst += sizeof(version);
        std::memcpy(dst, &width, sizeof(width));
        dst += sizeof(width);
        std::memcpy(dst, pending_h[seq_id].data(), row_size);
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            return true;
        }

        constexpr uint32_t expected_magic   = 0x3150544d; // MTP1
        constexpr uint32_t expected_version = 1;
        constexpr size_t header_size = sizeof(uint32_t) * 3;
        if (data.size() != header_size + (size_t) n_embd * sizeof(float)) {
            return false;
        }

        uint32_t magic;
        uint32_t version;
        uint32_t width;
        std::memcpy(&magic,   data.data(),                      sizeof(magic));
        std::memcpy(&version, data.data() + sizeof(uint32_t),   sizeof(version));
        std::memcpy(&width,   data.data() + sizeof(uint32_t)*2, sizeof(width));
        return magic == expected_magic && version == expected_version && width == uint32_t(n_embd);
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!validate_state(seq_id, data)) {
            return false;
        }

        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        i_last[seq_id] = -1;
        if (data.empty()) {
            std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
            return true;
        }

        constexpr size_t header_size = sizeof(uint32_t) * 3;
        std::memcpy(pending_h[seq_id].data(), data.data() + header_size,
                (size_t) n_embd * sizeof(float));
        return true;
    }

};

// Backend-owned Qwen3.8 MTP accelerator.  This class intentionally lives next
// to the native implementation: it consumes the same target h_nextn rows and
// participates in the same speculative/checkpoint state machine, but resolves
// all device work through the selected dynamic ggml backend.
struct common_speculative_impl_draft_mtp_hip : public common_speculative_impl {
    common_params_speculative_draft params;

    const ggml_spec_accel_api * api = nullptr;
    ggml_spec_accel_runtime_t runtime = nullptr;
    std::vector<ggml_spec_accel_seq_t> sequences;

    int32_t n_embd = 0;
    int32_t n_embd_inp = 0;

    std::vector<std::vector<float>> pending_h;
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;
    std::vector<int32_t> verify_pos_begin;
    std::vector<bool> disabled;

    struct checkpoint_state {
        uint32_t magic;
        uint32_t version;
        uint32_t width;
        uint32_t reserved;
        int32_t logical_pos;
        int32_t coverage_begin;
        int32_t coverage_end;
        uint32_t record_count;
        uint32_t reserved_2;
        uint64_t generation;
    };

    struct durable_state_header {
        uint32_t magic;
        uint32_t version;
        uint32_t width;
        uint32_t reserved;
        uint64_t backend_size;
    };

    static constexpr uint32_t state_magic = 0x484d544d; // MTMH
    static constexpr uint32_t state_version = 2;
    static constexpr uint32_t durable_magic = 0x4450544d; // MTPD
    static constexpr uint32_t durable_version = 1;

    static int32_t device_ordinal(const char * name) {
        if (name == nullptr) {
            return 0;
        }
        const std::string value(name);
        size_t begin = value.size();
        while (begin > 0 && std::isdigit(static_cast<unsigned char>(value[begin - 1]))) {
            --begin;
        }
        if (begin == value.size()) {
            return 0;
        }
        try {
            return std::stoi(value.substr(begin));
        } catch (...) {
            return 0;
        }
    }

    [[noreturn]] void startup_error(const std::string & message) const {
        throw std::runtime_error("HIP speculative accelerator: " + message);
    }

    std::string backend_error() const {
        if (api != nullptr && api->last_error != nullptr) {
            const char * message = api->last_error(runtime);
            if (message != nullptr && message[0] != '\0') {
                return message;
            }
        }
        return "backend operation failed";
    }

    void disable_sequence(llama_seq_id seq_id, const char * operation, int32_t status) {
        if (!disabled[seq_id]) {
            SPC_WRN("HIP MTP disabled for seq_id=%d after %s failed (status=%d): %s; "
                    "ngram and normal generation remain available\n",
                    (int) seq_id, operation, status, backend_error().c_str());
        }
        disabled[seq_id] = true;
    }

    void release_backend() noexcept {
        if (api == nullptr) {
            return;
        }
        for (auto & sequence : sequences) {
            if (sequence != nullptr) {
                api->sequence_free(sequence);
                sequence = nullptr;
            }
        }
        if (runtime != nullptr) {
            api->runtime_free(runtime);
            runtime = nullptr;
        }
    }

    common_speculative_impl_draft_mtp_hip(
            const common_params_speculative & all_params,
            uint32_t n_seq)
    try : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq,
                std::min(all_params.draft.n_max, 8))
        , params(all_params.draft) {
        if (params.ctx_tgt == nullptr) {
            startup_error("target context is unavailable");
        }
        if (params.accelerator_model.empty()) {
            startup_error("--spec-draft-accelerator-model is required when hip is selected");
        }
        if (params.target_model_path.empty()) {
            startup_error("target model path is unavailable for artifact identity validation");
        }
        if (params.p_min > 0.0f) {
            startup_error("confidence-threshold MTP is not supported; use --spec-draft-p-min 0");
        }

        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        llama_spec_accel_model_desc model_desc {};
        if (!llama_model_get_spec_accel_desc(model_tgt, &model_desc) ||
                model_desc.n_layer_nextn != 1 || !model_desc.is_mrope) {
            startup_error("target is not a supported single-NextN M-RoPE model");
        }
        n_embd = model_desc.n_embd;
        n_embd_inp = llama_model_n_embd(model_tgt);

        ggml_backend_dev_t selected = nullptr;
        auto try_device = [&](ggml_backend_dev_t device) {
            if (device == nullptr) {
                return false;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
            if (reg == nullptr) {
                return false;
            }
            auto getter = reinterpret_cast<ggml_backend_spec_accel_get_api_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_spec_accel_get_api"));
            if (getter == nullptr) {
                return false;
            }
            const ggml_spec_accel_api * candidate = getter(GGML_SPEC_ACCEL_ABI_VERSION);
            if (candidate == nullptr || candidate->abi_version != GGML_SPEC_ACCEL_ABI_VERSION ||
                    candidate->struct_size < sizeof(ggml_spec_accel_api)) {
                return false;
            }
            selected = device;
            api = candidate;
            return true;
        };

        for (ggml_backend_dev_t device : params.devices) {
            if (try_device(device)) {
                break;
            }
        }
        if (selected == nullptr && params.devices.empty()) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                if (try_device(ggml_backend_dev_get(i))) {
                    break;
                }
            }
        }
        if (selected == nullptr || api == nullptr) {
            startup_error("the selected draft device does not expose the accelerator ABI; "
                    "use a build with GGML_HIP_SPEC_ACCEL=ON and select the gfx1100 ROCm entry "
                    "reported by --list-devices (for example ROCm1)");
        }

        uint32_t cache_type = 0;
        if (params.cache_type_k == GGML_TYPE_F16 && params.cache_type_v == GGML_TYPE_F16) {
            cache_type = GGML_SPEC_ACCEL_CACHE_F16;
        } else if (params.cache_type_k == GGML_TYPE_Q8_0 && params.cache_type_v == GGML_TYPE_Q8_0) {
            cache_type = GGML_SPEC_ACCEL_CACHE_Q8_0;
        } else {
            startup_error("only homogeneous F16/F16 and Q8_0/Q8_0 private draft caches are supported; "
                    "the target model KV-cache selection is independent");
        }

        const char * selected_name = ggml_backend_dev_name(selected);
        ggml_spec_accel_descriptor descriptor {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = GGML_SPEC_ACCEL_KIND_MTP;
        descriptor.artifact_path = params.accelerator_model.c_str();
        descriptor.device_name = selected_name;
        descriptor.device_ordinal = device_ordinal(selected_name);
        descriptor.n_embd = model_desc.n_embd;
        descriptor.n_vocab = model_desc.n_vocab;
        descriptor.n_head = model_desc.n_head;
        descriptor.n_head_kv = model_desc.n_head_kv;
        descriptor.head_dim = model_desc.head_dim;
        descriptor.n_rot = model_desc.n_rot;
        descriptor.n_ctx = llama_n_ctx(params.ctx_tgt);
        descriptor.n_seq_max = n_seq;
        descriptor.cache_type = cache_type;
        descriptor.target_model_path = params.target_model_path.c_str();

        ggml_spec_accel_capabilities capabilities {};
        capabilities.struct_size = sizeof(capabilities);
        int32_t status = api->query_capabilities(&descriptor, &capabilities);
        if (status != GGML_SPEC_ACCEL_STATUS_OK ||
                !(capabilities.flags & GGML_SPEC_ACCEL_CAP_MTP) ||
                !(capabilities.flags & GGML_SPEC_ACCEL_CAP_TOKEN_EMBEDDINGS) ||
                !(capabilities.flags & GGML_SPEC_ACCEL_CAP_MROPE4) ||
                !(capabilities.flags & GGML_SPEC_ACCEL_CAP_MULTIMODAL_ROWS)) {
            startup_error(string_format(
                    "device/model/private-draft-cache combination is unsupported (status=%d)", status));
        }
        status = api->runtime_create(&descriptor, &runtime);
        if (status != GGML_SPEC_ACCEL_STATUS_OK || runtime == nullptr) {
            startup_error(string_format("runtime creation failed (status=%d): %s",
                    status, backend_error().c_str()));
        }

        sequences.assign(n_seq, nullptr);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            status = api->sequence_create(runtime, seq_id, &sequences[seq_id]);
            if (status != GGML_SPEC_ACCEL_STATUS_OK || sequences[seq_id] == nullptr) {
                startup_error(string_format("sequence %d creation failed (status=%d): %s",
                        (int) seq_id, status, backend_error().c_str()));
            }
        }

        llama_set_embeddings_nextn(params.ctx_tgt, true, false);
        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        verify_pos_begin.assign(n_seq, -1);
        disabled.assign(n_seq, false);
        this->params.n_max = std::min(this->params.n_max, 8);
        this->n_max = this->params.n_max;

        SPC_INF("HIP MTP accelerator enabled on %s: private-cache=%s, "
                "target-kv=independent/unrestricted, n_ctx=%u, slots=%u, artifact='%s'\n",
                selected_name ? selected_name : "ROCm",
                cache_type == GGML_SPEC_ACCEL_CACHE_Q8_0 ? "q8_0" : "f16",
                descriptor.n_ctx, n_seq, params.accelerator_model.c_str());
    } catch (...) {
        // A constructor that fails after creating the runtime or one of the
        // slot handles does not run the class destructor.  Release every
        // backend object here so an explicit accelerator startup error cannot
        // strand VRAM or a HIP stream before the server reports the failure.
        release_backend();
        throw;
    }

    ~common_speculative_impl_draft_mtp_hip() override {
        release_backend();
    }

    void begin(llama_seq_id, const llama_tokens &) override {
        // Prompt reuse/checkpoint restore owns sequence resets.  Resetting here
        // would discard a valid resident prefix selected by the server.
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }
        const bool has_tokens = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd != nullptr;
        if (has_tokens == has_embeddings || batch_in.pos == nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;
        std::vector<std::vector<int32_t>> indices(n_seq);
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (batch_in.n_seq_id[i] != 1 || batch_in.seq_id[i] == nullptr) {
                continue;
            }
            const llama_seq_id seq_id = batch_in.seq_id[i][0];
            if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
                indices[seq_id].push_back(i);
            }
        }

        const float * target_hidden = llama_get_embeddings_nextn(params.ctx_tgt);
        if (target_hidden == nullptr) {
            SPC_WRN("%s", "target h_nextn output is unavailable; HIP MTP skipped for this batch\n");
            return true;
        }
        const size_t row_bytes = size_t(n_embd) * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const auto & rows = indices[seq_id];
            if (rows.empty()) {
                continue;
            }
            const uint32_t count = uint32_t(rows.size());
            verify_h_rows[seq_id] = count;
            verify_pos_begin[seq_id] = batch_in.pos[rows.front()];
            verify_h[seq_id].resize(size_t(count) * n_embd);

            std::vector<float> hidden(size_t(count) * n_embd);
            std::vector<int32_t> positions(size_t(count) * 4);
            std::vector<int32_t> tokens;
            std::vector<float> embeddings;
            if (has_tokens) {
                tokens.resize(count);
            } else {
                embeddings.resize(size_t(count) * n_embd_inp);
            }

            for (uint32_t row = 0; row < count; ++row) {
                const int32_t source = rows[row];
                const float * h_row = target_hidden + size_t(source) * n_embd;
                std::memcpy(verify_h[seq_id].data() + size_t(row) * n_embd, h_row, row_bytes);
                const float * h_prev = row == 0 ? pending_h[seq_id].data() :
                        target_hidden + size_t(rows[row - 1]) * n_embd;
                std::memcpy(hidden.data() + size_t(row) * n_embd, h_prev, row_bytes);
                for (uint32_t plane = 0; plane < 4; ++plane) {
                    positions[size_t(plane) * count + row] =
                            has_tokens ? batch_in.pos[source] :
                                batch_in.pos[size_t(plane) * n_tokens + source];
                }
                if (has_tokens) {
                    tokens[row] = batch_in.token[source];
                } else {
                    std::memcpy(embeddings.data() + size_t(row) * n_embd_inp,
                            batch_in.embd + size_t(source) * n_embd_inp,
                            size_t(n_embd_inp) * sizeof(float));
                }
            }

            if (!disabled[seq_id]) {
                ggml_spec_accel_mtp_catchup catchup {};
                catchup.struct_size = sizeof(catchup);
                catchup.count = count;
                catchup.tokens = has_tokens ? tokens.data() : nullptr;
                catchup.token_embeddings = has_embeddings ? embeddings.data() : nullptr;
                catchup.hidden_rows = hidden.data();
                catchup.positions = positions.data();
                catchup.token_embedding_width = n_embd_inp;
                catchup.hidden_width = n_embd;
                catchup.position_planes = 4;
                const int32_t status = api->mtp_catchup(sequences[seq_id], &catchup);
                if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                    disable_sequence(seq_id, "KV-only catch-up", status);
                }
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + size_t(count - 1) * n_embd, row_bytes);
        }
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        std::array<llama_token, 8> output {};
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting || disabled[seq_id]) {
                continue;
            }
            int32_t limit = params.n_max;
            if (dp.n_max >= 0) {
                limit = std::min(limit, dp.n_max);
            }
            limit = std::min(limit, 8);
            if (limit <= 0 || limit < params.n_min) {
                continue;
            }

            ggml_spec_accel_mtp_draft request {};
            request.struct_size = sizeof(request);
            request.last_token = dp.id_last;
            request.past_records = dp.n_past;
            request.next_position = dp.pos_next;
            request.hidden = pending_h[seq_id].data();
            request.hidden_width = n_embd;
            request.max_draft = limit;
            request.output_ids = output.data();
            request.output_capacity = output.size();
            const int32_t status = api->mtp_draft(sequences[seq_id], &request);
            if (status == GGML_SPEC_ACCEL_STATUS_NO_COVERAGE ||
                    status == GGML_SPEC_ACCEL_STATUS_UNAVAILABLE) {
                continue;
            }
            if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                disable_sequence(seq_id, "draft", status);
                continue;
            }
            dp.result->assign(output.begin(), output.begin() + request.output_count);
            if (dp.result->size() < size_t(params.n_min)) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        const int32_t rows = verify_h_rows[seq_id];
        if (rows <= 0) {
            return;
        }
        const int32_t accepted_row = std::min<int32_t>(n_accepted, rows - 1);
        std::memcpy(pending_h[seq_id].data(),
                verify_h[seq_id].data() + size_t(accepted_row) * n_embd,
                size_t(n_embd) * sizeof(float));
        if (!disabled[seq_id] && verify_pos_begin[seq_id] >= 0) {
            const int32_t keep_end = verify_pos_begin[seq_id] + accepted_row + 1;
            const int32_t status = api->sequence_remove_suffix(sequences[seq_id], keep_end);
            if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                disable_sequence(seq_id, "accepted-suffix truncation", status);
            }
        }
    }

    bool decode_state(llama_seq_id seq_id, const std::vector<uint8_t> & data,
            checkpoint_state & state, const float *& pending) const {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq ||
                data.size() != sizeof(checkpoint_state) + size_t(n_embd) * sizeof(float)) {
            return false;
        }
        std::memcpy(&state, data.data(), sizeof(state));
        if (state.magic != state_magic || state.version != state_version ||
                state.width != uint32_t(n_embd) || state.reserved != 0 || state.reserved_2 != 0 ||
                state.logical_pos < 0 || state.coverage_begin < 0 ||
                state.coverage_end < state.coverage_begin || state.logical_pos > state.coverage_end ||
                state.record_count > uint32_t(llama_n_ctx(params.ctx_tgt))) {
            return false;
        }
        pending = reinterpret_cast<const float *>(data.data() + sizeof(state));
        return true;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        if (api->sequence_plan(sequences[seq_id], &plan) != GGML_SPEC_ACCEL_STATUS_OK) {
            return false;
        }
        const checkpoint_state state {
            state_magic, state_version, uint32_t(n_embd), 0,
            plan.logical_pos, plan.coverage_begin, plan.coverage_end,
            plan.record_count, 0, plan.generation,
        };
        data.resize(sizeof(state) + size_t(n_embd) * sizeof(float));
        std::memcpy(data.data(), &state, sizeof(state));
        std::memcpy(data.data() + sizeof(state), pending_h[seq_id].data(),
                size_t(n_embd) * sizeof(float));
        return true;
    }

    bool validate_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        if (data.empty()) {
            return true;
        }
        checkpoint_state state {};
        const float * pending = nullptr;
        if (!decode_state(seq_id, data, state, pending)) {
            return false;
        }
        ggml_spec_accel_checkpoint_desc descriptor {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.logical_pos = state.logical_pos;
        descriptor.coverage_begin = state.coverage_begin;
        descriptor.coverage_end = state.coverage_end;
        descriptor.record_count = state.record_count;
        descriptor.reserved = 0;
        descriptor.generation = state.generation;
        descriptor.pending_hidden = pending;
        descriptor.pending_hidden_count = n_embd;
        ggml_spec_accel_checkpoint_t transaction = nullptr;
        const int32_t status = api->checkpoint_validate(
                sequences[seq_id], &descriptor, &transaction);
        if (transaction != nullptr) {
            api->checkpoint_free(transaction);
        }
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        verify_pos_begin[seq_id] = -1;
        if (data.empty()) {
            const int32_t status = api->sequence_reset(sequences[seq_id]);
            if (status != GGML_SPEC_ACCEL_STATUS_OK) {
                return false;
            }
            std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
            disabled[seq_id] = false;
            return true;
        }

        checkpoint_state state {};
        const float * pending = nullptr;
        if (!decode_state(seq_id, data, state, pending)) {
            return false;
        }
        ggml_spec_accel_checkpoint_desc descriptor {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.logical_pos = state.logical_pos;
        descriptor.coverage_begin = state.coverage_begin;
        descriptor.coverage_end = state.coverage_end;
        descriptor.record_count = state.record_count;
        descriptor.reserved = 0;
        descriptor.generation = state.generation;
        descriptor.pending_hidden = pending;
        descriptor.pending_hidden_count = n_embd;
        ggml_spec_accel_checkpoint_t transaction = nullptr;
        int32_t status = api->checkpoint_validate(sequences[seq_id], &descriptor, &transaction);
        if (status != GGML_SPEC_ACCEL_STATUS_OK || transaction == nullptr) {
            return false;
        }
        status = api->checkpoint_commit(transaction);
        api->checkpoint_free(transaction);
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            return false;
        }
        std::memcpy(pending_h[seq_id].data(), pending, size_t(n_embd) * sizeof(float));
        disabled[seq_id] = false;
        return true;
    }

    bool decode_durable_state(
            llama_seq_id seq_id,
            const std::vector<uint8_t> & data,
            durable_state_header & header,
            const float *& pending,
            const uint8_t *& backend) const {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq ||
                data.size() < sizeof(header) + size_t(n_embd) * sizeof(float)) {
            return false;
        }
        std::memcpy(&header, data.data(), sizeof(header));
        const size_t prefix = sizeof(header) + size_t(n_embd) * sizeof(float);
        if (header.magic != durable_magic || header.version != durable_version ||
                header.width != uint32_t(n_embd) || header.reserved != 0 ||
                header.backend_size != data.size() - prefix) {
            return false;
        }
        pending = reinterpret_cast<const float *>(data.data() + sizeof(header));
        backend = data.data() + prefix;
        return true;
    }

    bool get_state_durable(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        data.clear();
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_blob blob {};
        blob.struct_size = sizeof(blob);
        const int32_t status = api->state_export(sequences[seq_id], &blob);
        if (status != GGML_SPEC_ACCEL_STATUS_OK || blob.data == nullptr || blob.size == 0 ||
                blob.size > std::numeric_limits<size_t>::max() - sizeof(durable_state_header) -
                    size_t(n_embd) * sizeof(float)) {
            if (blob.data != nullptr) {
                api->blob_free(&blob);
            }
            return false;
        }
        const durable_state_header header {
            durable_magic, durable_version, uint32_t(n_embd), 0, blob.size,
        };
        try {
            data.resize(sizeof(header) + size_t(n_embd) * sizeof(float) + blob.size);
            std::memcpy(data.data(), &header, sizeof(header));
            std::memcpy(data.data() + sizeof(header), pending_h[seq_id].data(),
                    size_t(n_embd) * sizeof(float));
            std::memcpy(data.data() + sizeof(header) + size_t(n_embd) * sizeof(float),
                    blob.data, blob.size);
        } catch (...) {
            api->blob_free(&blob);
            throw;
        }
        api->blob_free(&blob);
        return true;
    }

    bool validate_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) const override {
        if (data.empty()) {
            return seq_id >= 0 && seq_id < (llama_seq_id) n_seq;
        }
        durable_state_header header {};
        const float * pending = nullptr;
        const uint8_t * backend = nullptr;
        return decode_durable_state(seq_id, data, header, pending, backend) &&
                api->state_validate(sequences[seq_id], backend, size_t(header.backend_size)) ==
                    GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool set_state_durable(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (data.empty()) {
            return set_state(seq_id, data);
        }
        durable_state_header header {};
        const float * pending = nullptr;
        const uint8_t * backend = nullptr;
        if (!decode_durable_state(seq_id, data, header, pending, backend) ||
                api->state_import(sequences[seq_id], backend, size_t(header.backend_size)) !=
                    GGML_SPEC_ACCEL_STATUS_OK) {
            return false;
        }
        std::memcpy(pending_h[seq_id].data(), pending, size_t(n_embd) * sizeof(float));
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        verify_pos_begin[seq_id] = -1;
        disabled[seq_id] = false;
        return true;
    }

    void cancel_state_durable(llama_seq_id seq_id) override {
        if (seq_id >= 0 && seq_id < (llama_seq_id) n_seq) {
            api->state_cancel(sequences[seq_id]);
        }
    }

    bool sequence_copy(llama_seq_id source, llama_seq_id destination) override {
        if (source < 0 || destination < 0 || source >= (llama_seq_id) n_seq ||
                destination >= (llama_seq_id) n_seq) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        int32_t status = api->sequence_plan(sequences[source], &plan);
        if (status == GGML_SPEC_ACCEL_STATUS_OK && plan.coverage_begin == plan.coverage_end) {
            status = api->sequence_invalidate(sequences[destination], plan.logical_pos);
        } else if (status == GGML_SPEC_ACCEL_STATUS_OK) {
            status = api->sequence_copy(sequences[source], sequences[destination],
                    plan.coverage_begin, plan.coverage_end);
        }
        pending_h[destination] = pending_h[source];
        verify_h[destination].clear();
        verify_h_rows[destination] = 0;
        verify_pos_begin[destination] = -1;
        disabled[destination] = disabled[source] || status != GGML_SPEC_ACCEL_STATUS_OK;
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool sequence_remove_suffix(llama_seq_id seq_id, int32_t pos) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pos < 0) {
            return false;
        }
        const int32_t status = api->sequence_remove_suffix(sequences[seq_id], pos);
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            disable_sequence(seq_id, "sequence suffix removal", status);
        }
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }

    bool sequence_shift(llama_seq_id seq_id, int32_t begin, int32_t end, int32_t delta) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || begin < 0 || end < begin || delta == 0) {
            return false;
        }
        ggml_spec_accel_sequence_plan plan {};
        plan.struct_size = sizeof(plan);
        int32_t status = api->sequence_plan(sequences[seq_id], &plan);
        const int32_t resident_begin = std::max(begin, plan.coverage_begin);
        const int32_t resident_end = std::min(end, plan.coverage_end);
        if (status == GGML_SPEC_ACCEL_STATUS_OK && resident_begin < resident_end) {
            status = api->sequence_shift(
                    sequences[seq_id], resident_begin, resident_end, delta);
        } else if (status == GGML_SPEC_ACCEL_STATUS_OK) {
            const int64_t shifted = int64_t(plan.logical_pos) + delta;
            status = shifted >= 0 && shifted <= std::numeric_limits<int32_t>::max() ?
                    api->sequence_invalidate(sequences[seq_id], int32_t(shifted)) :
                    GGML_SPEC_ACCEL_STATUS_INVALID;
        }
        if (status != GGML_SPEC_ACCEL_STATUS_OK) {
            disable_sequence(seq_id, "sequence position shift", status);
        }
        return status == GGML_SPEC_ACCEL_STATUS_OK;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

void common_ngram_mod_adaptive_begin(
        common_ngram_mod_adaptive_state & state,
        size_t configured_min,
        size_t configured_max) {
    configured_min = std::min(configured_min, configured_max);
    state.n_max = std::min(
            configured_max, std::max(configured_min, size_t(16)));
    state.n_bad = 0;
    state.n_good = 0;
    state.cooldown = 0;
}

bool common_ngram_mod_adaptive_should_draft(
        common_ngram_mod_adaptive_state & state,
        bool costly_rollback) {
    if (!costly_rollback || state.cooldown == 0) {
        return true;
    }
    --state.cooldown;
    return false;
}

void common_ngram_mod_adaptive_accept(
        common_ngram_mod_adaptive_state & state,
        size_t configured_min,
        size_t configured_max,
        size_t proposed,
        size_t accepted,
        uint32_t native_rollback,
        bool costly_rollback) {
    if (!costly_rollback || proposed == 0) {
        return;
    }

    configured_min = std::min(configured_min, configured_max);
    accepted = std::min(accepted, proposed);
    const size_t rejected = proposed - accepted;
    const bool checkpoint_miss = rejected > size_t(native_rollback);

    if (rejected == 0) {
        state.cooldown = 0;

        // A short n-gram can end before the adaptive horizon. It is profitable,
        // but it is not evidence that a larger verifier batch will be. Require
        // two complete probes at the current horizon before adding one bucket.
        if (proposed >= state.n_max) {
            state.n_good = std::min<uint32_t>(state.n_good + 1, 2);
            if (state.n_good >= 2) {
                state.n_good = 0;
                if (state.n_bad > 0) {
                    --state.n_bad;
                }
                state.n_max = std::min(configured_max, state.n_max + size_t(8));
            }
        } else {
            state.n_good = 0;
        }
        return;
    }

    state.n_good = 0;
    if (!checkpoint_miss) {
        // Rejections inside the native snapshot horizon do not restore or
        // replay a checkpoint, so preserve the profitable horizon.
        state.cooldown = 0;
        return;
    }

    // Every checkpoint-backed rejection is expensive, even when half or more
    // of the proposal matched. Retry no farther than the observed complete
    // eight-token bucket and yield increasingly often if such misses recur.
    state.n_bad = std::min<uint32_t>(state.n_bad + 1, 4);
    const size_t observed = (accepted / 8) * 8;
    state.n_max = std::max(
            configured_min,
            std::min(state.n_max, std::max(size_t(1), observed)));

    uint32_t cooldown = uint32_t(1) << state.n_bad;
    if (accepted == 0) {
        cooldown *= 2;
    }
    state.cooldown = std::min<uint32_t>(cooldown, 16);
}

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // ngram-mod can be very profitable on an exact repeated span, but a
        // long miss is expensive on recurrent targets: restoring their state
        // may require replaying the accepted prefix.  Start at a useful batch
        // size, promote after sustained complete matches, and back off locally
        // after a checkpoint-backed miss instead of clearing the shared table.
        common_ngram_mod_adaptive_state adaptive;
        int n_low = 0;
        bool costly_rollback = false;
        uint32_t native_rollback = UINT32_MAX;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;
        sinfo.n_low = 0;
        sinfo.costly_rollback = false;
        sinfo.native_rollback = UINT32_MAX;

        const size_t configured_max = std::max(0, params.n_max);
        const size_t configured_min = std::min(
                configured_max, size_t(std::max(1, params.n_min)));
        common_ngram_mod_adaptive_begin(
                sinfo.adaptive, configured_min, configured_max);

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        sinfo.costly_rollback = dparams.costly_rollback;
        sinfo.native_rollback = dparams.native_rollback;

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        if (!common_ngram_mod_adaptive_should_draft(
                    sinfo.adaptive, sinfo.costly_rollback)) {
            SPC_TRC("ngram_mod seq=%d cooling down, remaining=%u adaptive-max=%zu\n",
                    (int) seq_id, sinfo.adaptive.cooldown, sinfo.adaptive.n_max);
            return;
        }

        size_t n_limit = size_t(std::max(0, params.n_max));
        if (dparams.n_max > 0) {
            n_limit = std::min(n_limit, size_t(dparams.n_max));
        }
        if (sinfo.costly_rollback && sinfo.adaptive.n_max > 0) {
            n_limit = std::min(n_limit, sinfo.adaptive.n_max);
        }
        if (n_limit == 0) {
            return;
        }

        const size_t n_required = std::min(
                n_limit, size_t(std::max(0, params.n_min)));

        result.resize(n + n_limit);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (size_t i = 0; i < n_limit; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < n_required) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        if (sinfo.n_draft_last == 0) {
            return;
        }

        const size_t proposed = sinfo.n_draft_last;
        const size_t accepted = std::min<size_t>(n_accepted, proposed);

        if (!sinfo.costly_rollback) {
            const double f_acc = double(accepted) / double(proposed);
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }
                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
            return;
        }

        const size_t rejected = proposed - accepted;
        const size_t configured_max = size_t(std::max(0, params.n_max));
        const size_t configured_min = std::min(
                configured_max, size_t(std::max(1, params.n_min)));
        const size_t old_max = sinfo.adaptive.n_max;
        const uint32_t old_bad = sinfo.adaptive.n_bad;
        const bool checkpoint_miss = rejected > size_t(sinfo.native_rollback);

        common_ngram_mod_adaptive_accept(
                sinfo.adaptive, configured_min, configured_max,
                proposed, accepted, sinfo.native_rollback,
                sinfo.costly_rollback);

        if (verbose || sinfo.adaptive.n_max != old_max ||
                sinfo.adaptive.n_bad != old_bad || sinfo.adaptive.cooldown > 0) {
            SPC_TRC("ngram_mod seq=%d accepted=%zu/%zu checkpoint-miss=%s "
                    "adaptive-max=%zu->%zu bad=%u cooldown=%u\n",
                    (int) seq_id, accepted, proposed,
                    checkpoint_miss ? "yes" : "no", old_max, sinfo.adaptive.n_max,
                    sinfo.adaptive.n_bad, sinfo.adaptive.cooldown);
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

bool common_speculative_dflash_adaptive_dm_supported(int32_t selector_top_k) {
    return selector_top_k <= 0;
}

bool common_speculative_adaptive_dm_supported(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (const auto & impl : spec->impls) {
        if (impl->adaptive_dm_supported()) {
            return true;
        }
    }

    return false;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // A tensor split is a topology property of the model being loaded, not
        // a global property that can be inherited from the target.  In
        // particular, a single explicitly selected draft device cannot form a
        // tensor-split meta device.  Keep upstream's split behavior for true
        // multi-device drafts, but collapse the degenerate topology before
        // model fitting and buffer placement see it.
        const size_t n_draft_devices = std::count_if(
            result.devices.begin(), result.devices.end(), [](ggml_backend_dev_t dev) { return dev != nullptr; });
        if (!result.devices.empty() && n_draft_devices <= 1) {
            result.split_mode = LLAMA_SPLIT_MODE_NONE;
            std::fill(std::begin(result.tensor_split), std::end(result.tensor_split), 0.0f);
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.kv_tail_tokens = "0";
    result.kv_tail_type   = GGML_TYPE_F16;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    const bool spec_accel_hip = params.speculative.draft.accelerator ==
            COMMON_SPECULATIVE_DRAFT_ACCELERATOR_HIP;

    // A backend-owned accelerator does not allocate a native llama draft
    // model/context.  It consumes target h_nextn rows through common and owns
    // the compact weights plus per-slot cache inside the dynamic HIP backend.
    if (spec_accel_hip) {
        return;
    }

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.kv_tail_rollback_tokens = 0;
    cparams.ctx_other = ctx_tgt;
    cparams.kv_tail_tokens = 0;
    cparams.kv_tail_type   = GGML_TYPE_F16;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        const bool has_external_accelerator =
                params.draft.accelerator == COMMON_SPECULATIVE_DRAFT_ACCELERATOR_HIP;
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
                params.draft.ctx_dft != nullptr || has_external_accelerator);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
                params.draft.ctx_dft != nullptr || has_external_accelerator);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                if (config.params.draft.accelerator == COMMON_SPECULATIVE_DRAFT_ACCELERATOR_HIP) {
                    impls.push_back(std::make_unique<common_speculative_impl_draft_mtp_hip>(
                            config.params, n_seq));
                } else {
                    impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(
                            config.params, n_seq));
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                if (config.params.draft.accelerator == COMMON_SPECULATIVE_DRAFT_ACCELERATOR_HIP) {
                    impls.push_back(std::make_unique<common_speculative_impl_draft_dflash_hip>(
                            config.params, n_seq));
                } else {
                    impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                            config.params, n_seq));
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

bool common_speculative_seq_cp(
        common_speculative * spec,
        llama_seq_id source,
        llama_seq_id destination) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->sequence_copy(source, destination) && result;
    }
    return result;
}

bool common_speculative_seq_rm_suffix(
        common_speculative * spec,
        llama_seq_id seq_id,
        int32_t pos) {
    if (spec == nullptr) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->sequence_remove_suffix(seq_id, pos) && result;
    }
    return result;
}

bool common_speculative_seq_add(
        common_speculative * spec,
        llama_seq_id seq_id,
        int32_t begin,
        int32_t end,
        int32_t delta) {
    if (spec == nullptr || delta == 0) {
        return true;
    }
    bool result = true;
    for (auto & impl : spec->impls) {
        result = impl->sequence_shift(seq_id, begin, end, delta) && result;
    }
    return result;
}

static bool common_speculative_get_state_internal(
        common_speculative * spec,
        llama_seq_id seq_id,
        std::vector<uint8_t> & data,
        bool durable) {
    data.clear();
    if (spec == nullptr) {
        return false;
    }

    struct entry {
        uint32_t type;
        std::vector<uint8_t> payload;
    };
    std::vector<entry> entries;
    for (auto & impl : spec->impls) {
        std::vector<uint8_t> payload;
        if (durable ? impl->get_state_durable(seq_id, payload) : impl->get_state(seq_id, payload)) {
            entries.push_back({ uint32_t(impl->type), std::move(payload) });
        }
    }
    if (entries.empty()) {
        return false;
    }

    constexpr uint32_t magic = 0x43455053; // SPEC
    uint64_t checksum = 1469598103934665603ULL;
    const auto hash_bytes = [&](const void * ptr, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            checksum = (checksum ^ bytes[i])*1099511628211ULL;
        }
    };

    // Keep the original single-state envelope byte-for-byte compatible with
    // existing RAM prompt caches and checkpoints.
    if (entries.size() == 1) {
        constexpr uint32_t version = 1;
        const uint32_t type = entries[0].type;
        const uint64_t payload_size = entries[0].payload.size();
        hash_bytes(&magic, sizeof(magic));
        hash_bytes(&version, sizeof(version));
        hash_bytes(&type, sizeof(type));
        hash_bytes(&seq_id, sizeof(seq_id));
        hash_bytes(&payload_size, sizeof(payload_size));
        hash_bytes(entries[0].payload.data(), entries[0].payload.size());
        data.resize(sizeof(magic) + sizeof(version) + sizeof(type) + sizeof(seq_id) +
                sizeof(payload_size) + sizeof(checksum) + entries[0].payload.size());
        uint8_t * dst = data.data();
        const auto append = [&](const auto & value) {
            std::memcpy(dst, &value, sizeof(value));
            dst += sizeof(value);
        };
        append(magic);
        append(version);
        append(type);
        append(seq_id);
        append(payload_size);
        append(checksum);
        std::memcpy(dst, entries[0].payload.data(), entries[0].payload.size());
        return true;
    }

    constexpr uint32_t version = 2;
    constexpr uint32_t reserved = 0;
    const uint32_t entry_count = uint32_t(entries.size());
    uint64_t payload_size = 0;
    for (const auto & value : entries) {
        constexpr uint64_t entry_header_size = sizeof(uint32_t)*2 + sizeof(uint64_t);
        if (payload_size > std::numeric_limits<uint64_t>::max() - entry_header_size ||
                value.payload.size() >
                    std::numeric_limits<uint64_t>::max() - entry_header_size - payload_size) {
            return false;
        }
        payload_size += entry_header_size + value.payload.size();
    }
    constexpr size_t header_size = sizeof(uint32_t)*3 + sizeof(llama_seq_id) + sizeof(uint64_t)*2;
    if (payload_size > std::numeric_limits<size_t>::max() - header_size) {
        return false;
    }
    hash_bytes(&magic, sizeof(magic));
    hash_bytes(&version, sizeof(version));
    hash_bytes(&seq_id, sizeof(seq_id));
    hash_bytes(&entry_count, sizeof(entry_count));
    hash_bytes(&payload_size, sizeof(payload_size));
    for (const auto & value : entries) {
        const uint64_t size = value.payload.size();
        hash_bytes(&value.type, sizeof(value.type));
        hash_bytes(&reserved, sizeof(reserved));
        hash_bytes(&size, sizeof(size));
        hash_bytes(value.payload.data(), value.payload.size());
    }
    data.resize(header_size + size_t(payload_size));
    uint8_t * dst = data.data();
    const auto append = [&](const auto & value) {
        std::memcpy(dst, &value, sizeof(value));
        dst += sizeof(value);
    };
    append(magic);
    append(version);
    append(seq_id);
    append(entry_count);
    append(payload_size);
    append(checksum);
    for (const auto & value : entries) {
        const uint64_t size = value.payload.size();
        append(value.type);
        append(reserved);
        append(size);
        std::memcpy(dst, value.payload.data(), value.payload.size());
        dst += value.payload.size();
    }
    return true;
}

bool common_speculative_get_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        std::vector<uint8_t> & data) {
    return common_speculative_get_state_internal(spec, seq_id, data, false);
}

bool common_speculative_get_state_durable(
        common_speculative * spec,
        llama_seq_id seq_id,
        std::vector<uint8_t> & data) {
    return common_speculative_get_state_internal(spec, seq_id, data, true);
}

namespace {
struct common_speculative_state_view {
    uint32_t type = 0;
    const uint8_t * payload = nullptr;
    size_t payload_size = 0;
};

bool common_speculative_parse_state_v1(
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t data_size,
        common_speculative_state_view & view) {
    constexpr size_t header_size = sizeof(uint32_t)*3 + sizeof(llama_seq_id) + sizeof(uint64_t)*2;
    if (data_size < header_size || data == nullptr) {
        return false;
    }
    const uint8_t * src = data;
    const auto read = [&](auto & value) {
        std::memcpy(&value, src, sizeof(value));
        src += sizeof(value);
    };
    uint32_t magic;
    uint32_t version;
    llama_seq_id saved_seq_id;
    uint64_t payload_size;
    uint64_t checksum;
    read(magic);
    read(version);
    read(view.type);
    read(saved_seq_id);
    read(payload_size);
    read(checksum);
    // The saved sequence ID is provenance, not ownership.  RAM snapshots can
    // be restored into a different server slot, so the envelope must not bind
    // otherwise portable implementation state to its source slot number.
    if (magic != 0x43455053 || version != 1 || saved_seq_id < 0 || seq_id < 0 ||
            payload_size != data_size - header_size) {
        return false;
    }
    uint64_t actual_checksum = 1469598103934665603ULL;
    const auto hash_bytes = [&](const void * ptr, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            actual_checksum = (actual_checksum ^ bytes[i])*1099511628211ULL;
        }
    };
    hash_bytes(&magic, sizeof(magic));
    hash_bytes(&version, sizeof(version));
    hash_bytes(&view.type, sizeof(view.type));
    hash_bytes(&saved_seq_id, sizeof(saved_seq_id));
    hash_bytes(&payload_size, sizeof(payload_size));
    for (size_t i = 0; i < payload_size; ++i) {
        actual_checksum = (actual_checksum ^ src[i])*1099511628211ULL;
    }
    if (actual_checksum != checksum) {
        return false;
    }
    view.payload = src;
    view.payload_size = size_t(payload_size);
    return true;
}

bool common_speculative_parse_states(
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t data_size,
        std::vector<common_speculative_state_view> & views) {
    views.clear();
    if (data == nullptr || data_size < sizeof(uint32_t)*2) {
        return false;
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    std::memcpy(&magic, data, sizeof(magic));
    std::memcpy(&version, data + sizeof(magic), sizeof(version));
    if (magic != 0x43455053) {
        return false;
    }
    if (version == 1) {
        common_speculative_state_view view;
        if (!common_speculative_parse_state_v1(seq_id, data, data_size, view)) {
            return false;
        }
        views.push_back(view);
        return true;
    }
    constexpr size_t header_size = sizeof(uint32_t)*3 + sizeof(llama_seq_id) + sizeof(uint64_t)*2;
    constexpr size_t entry_header_size = sizeof(uint32_t)*2 + sizeof(uint64_t);
    if (version != 2 || data_size < header_size) {
        return false;
    }
    const uint8_t * src = data;
    const auto read = [&](auto & value) {
        std::memcpy(&value, src, sizeof(value));
        src += sizeof(value);
    };
    llama_seq_id saved_seq_id;
    uint32_t entry_count;
    uint64_t payload_size;
    uint64_t checksum;
    read(magic);
    read(version);
    read(saved_seq_id);
    read(entry_count);
    read(payload_size);
    read(checksum);
    if (saved_seq_id < 0 || seq_id < 0 || entry_count == 0 ||
            payload_size != data_size - header_size || entry_count > 64) {
        return false;
    }
    uint64_t actual_checksum = 1469598103934665603ULL;
    const auto hash_bytes = [&](const void * ptr, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < size; ++i) {
            actual_checksum = (actual_checksum ^ bytes[i])*1099511628211ULL;
        }
    };
    hash_bytes(&magic, sizeof(magic));
    hash_bytes(&version, sizeof(version));
    hash_bytes(&saved_seq_id, sizeof(saved_seq_id));
    hash_bytes(&entry_count, sizeof(entry_count));
    hash_bytes(&payload_size, sizeof(payload_size));
    views.reserve(entry_count);
    for (uint32_t i = 0; i < entry_count; ++i) {
        if (size_t(data + data_size - src) < entry_header_size) {
            return false;
        }
        common_speculative_state_view view;
        uint32_t reserved;
        uint64_t size;
        read(view.type);
        read(reserved);
        read(size);
        if (reserved != 0 || size > uint64_t(data + data_size - src) ||
                std::any_of(views.begin(), views.end(), [&](const auto & prior) {
                    return prior.type == view.type;
                })) {
            return false;
        }
        hash_bytes(&view.type, sizeof(view.type));
        hash_bytes(&reserved, sizeof(reserved));
        hash_bytes(&size, sizeof(size));
        hash_bytes(src, size_t(size));
        view.payload = src;
        view.payload_size = size_t(size);
        views.push_back(view);
        src += size_t(size);
    }
    return src == data + data_size && actual_checksum == checksum;
}
}

bool common_speculative_validate_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return data.empty();
    }
    if (data.empty()) {
        return std::all_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
            return impl->validate_state(seq_id, data);
        });
    }

    std::vector<common_speculative_state_view> views;
    if (!common_speculative_parse_states(seq_id, data.data(), data.size(), views)) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        const auto match = std::find_if(views.begin(), views.end(), [&](const auto & view) {
            return uint32_t(impl->type) == view.type;
        });
        if (match == views.end()) {
            if (!impl->validate_state(seq_id, {})) {
                return false;
            }
            continue;
        }
        const std::vector<uint8_t> payload(match->payload, match->payload + match->payload_size);
        if (!impl->validate_state(seq_id, payload)) {
            return false;
        }
    }
    for (const auto & view : views) {
        if (std::none_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
                    return uint32_t(impl->type) == view.type;
                })) {
            return false;
        }
    }
    return true;
}

struct common_speculative_state_restore_plan {
    struct item {
        common_speculative_impl * impl = nullptr;
        std::vector<uint8_t> payload;
    };
    common_speculative * spec = nullptr;
    llama_seq_id seq_id = -1;
    std::vector<item> items;
    bool clear_all = false;
    bool durable = false;
    bool committed = false;

    ~common_speculative_state_restore_plan() {
        if (!durable || committed) {
            return;
        }
        for (auto & item : items) {
            if (item.impl != nullptr) {
                item.impl->cancel_state_durable(seq_id);
            }
        }
    }
};

static common_speculative_state_restore_plan * common_speculative_prepare_state_internal(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size,
        bool durable) {
    try {
        auto plan = std::make_unique<common_speculative_state_restore_plan>();
        plan->spec = spec;
        plan->seq_id = seq_id;
        plan->durable = durable;

        if (spec == nullptr) {
            return size == 0 ? plan.release() : nullptr;
        }
        if (size == 0) {
            const std::vector<uint8_t> empty;
            if (!std::all_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
                        return durable ? impl->validate_state_durable(seq_id, empty) :
                                impl->validate_state(seq_id, empty);
                    })) {
                return nullptr;
            }
            plan->clear_all = true;
            return plan.release();
        }

        std::vector<common_speculative_state_view> views;
        if (!common_speculative_parse_states(seq_id, data, size, views)) {
            return nullptr;
        }
        for (auto & impl : spec->impls) {
            const auto match = std::find_if(views.begin(), views.end(), [&](const auto & view) {
                return uint32_t(impl->type) == view.type;
            });
            common_speculative_state_restore_plan::item item;
            item.impl = impl.get();
            if (match != views.end()) {
                item.payload.assign(match->payload, match->payload + match->payload_size);
            }
            plan->items.push_back(std::move(item));
            auto & prepared = plan->items.back();
            if (durable ? !impl->validate_state_durable(seq_id, prepared.payload) :
                    !impl->validate_state(seq_id, prepared.payload)) {
                return nullptr;
            }
        }
        for (const auto & view : views) {
            if (std::none_of(spec->impls.begin(), spec->impls.end(), [&](const auto & impl) {
                        return uint32_t(impl->type) == view.type;
                    })) {
                return nullptr;
            }
        }
        return plan.release();
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
    return nullptr;
}

common_speculative_state_restore_plan * common_speculative_prepare_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size) {
    return common_speculative_prepare_state_internal(spec, seq_id, data, size, false);
}

common_speculative_state_restore_plan * common_speculative_prepare_state_durable(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size) {
    return common_speculative_prepare_state_internal(spec, seq_id, data, size, true);
}

void common_speculative_state_restore_plan_commit(common_speculative_state_restore_plan * plan) {
    GGML_ASSERT(plan != nullptr);
    if (plan->spec == nullptr) {
        return;
    }
    if (plan->clear_all) {
        const std::vector<uint8_t> empty;
        for (auto & impl : plan->spec->impls) {
            GGML_ASSERT(plan->durable ? impl->set_state_durable(plan->seq_id, empty) :
                    impl->set_state(plan->seq_id, empty));
        }
        return;
    }
    for (auto & item : plan->items) {
        GGML_ASSERT(item.impl != nullptr);
        GGML_ASSERT(plan->durable ? item.impl->set_state_durable(plan->seq_id, item.payload) :
                item.impl->set_state(plan->seq_id, item.payload));
    }
    plan->committed = true;
}

void common_speculative_state_restore_plan_free(common_speculative_state_restore_plan * plan) {
    delete plan;
}

bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    std::unique_ptr<common_speculative_state_restore_plan,
            decltype(&common_speculative_state_restore_plan_free)> plan(
        common_speculative_prepare_state(spec, seq_id, data.data(), data.size()),
        common_speculative_state_restore_plan_free);
    if (!plan) {
        return false;
    }
    common_speculative_state_restore_plan_commit(plan.get());
    return true;
}

common_speculative_type common_speculative_last_type(
        const common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 ||
            size_t(seq_id) >= spec->impl_last.size() ||
            spec->impl_last[seq_id] == nullptr) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }
    return spec->impl_last[seq_id]->type;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
