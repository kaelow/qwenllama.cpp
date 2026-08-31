#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// Bee's variable-depth adaptive draft-max is limited to the original DFlash path.
bool common_speculative_dflash_adaptive_dm_supported(int32_t selector_top_k);
bool common_speculative_adaptive_dm_supported(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    // The target can remove only a bounded suffix and may need checkpoint
    // replay after a rejection. Cheap/arbitrary rollback keeps the upstream
    // n-gram horizon behavior.
    bool costly_rollback = false;

    // Number of rejected suffix tokens the target can remove without restoring
    // and replaying a checkpoint. UINT32_MAX denotes arbitrary suffix removal.
    uint32_t native_rollback = UINT32_MAX;

    // Number of target input records already accepted.  This is normally the
    // same as pos_next, but multimodal M-RoPE batches can contain many image
    // embedding rows at one model position.
    int32_t     n_past;

    // Model position assigned to the next sampled/drafted text token.
    llama_pos   pos_next;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

// Profit guard for ngram-mod on checkpoint-bound targets. Kept as a small pure
// state machine so rejection/cooldown behavior can be regression-tested without
// loading a model.
struct common_ngram_mod_adaptive_state {
    size_t n_max = 0;
    uint32_t n_bad = 0;
    uint32_t n_good = 0;
    uint32_t cooldown = 0;
};

void common_ngram_mod_adaptive_begin(
        common_ngram_mod_adaptive_state & state,
        size_t configured_min,
        size_t configured_max);

bool common_ngram_mod_adaptive_should_draft(
        common_ngram_mod_adaptive_state & state,
        bool costly_rollback);

void common_ngram_mod_adaptive_accept(
        common_ngram_mod_adaptive_state & state,
        size_t configured_min,
        size_t configured_max,
        size_t proposed,
        size_t accepted,
        uint32_t native_rollback,
        bool costly_rollback);

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// Keep backend-owned speculative sequence state synchronized with the target
// and native draft memories during slot forks, suffix removal, and shifts.
bool common_speculative_seq_cp(
        common_speculative * spec,
        llama_seq_id source,
        llama_seq_id destination);
bool common_speculative_seq_rm_suffix(
        common_speculative * spec,
        llama_seq_id seq_id,
        int32_t pos);
bool common_speculative_seq_add(
        common_speculative * spec,
        llama_seq_id seq_id,
        int32_t begin,
        int32_t end,
        int32_t delta);

// Returns the implementation that produced the most recent draft for a
// sequence. Used for trace diagnostics and failure attribution only.
enum common_speculative_type common_speculative_last_type(
        const common_speculative * spec, llama_seq_id seq_id);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_validate_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// Self-contained state for persistent prompt caches.  Ordinary speculative
// checkpoints intentionally use the compact functions above.
bool common_speculative_get_state_durable(
        common_speculative * spec,
        llama_seq_id seq_id,
        std::vector<uint8_t> & data);

// Prepare validates and owns the decoded implementation payload. Commit has no
// remaining parsing or allocation and cannot fail.
struct common_speculative_state_restore_plan;
common_speculative_state_restore_plan * common_speculative_prepare_state(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size);
common_speculative_state_restore_plan * common_speculative_prepare_state_durable(
        common_speculative * spec,
        llama_seq_id seq_id,
        const uint8_t * data,
        size_t size);
void common_speculative_state_restore_plan_commit(common_speculative_state_restore_plan * plan);
void common_speculative_state_restore_plan_free(common_speculative_state_restore_plan * plan);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
