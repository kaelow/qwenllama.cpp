#pragma once

#include "ggml-spec-accel.h"

#ifdef __cplusplus
extern "C" {
#endif

const struct ggml_spec_accel_api * ggml_backend_spec_accel_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif
