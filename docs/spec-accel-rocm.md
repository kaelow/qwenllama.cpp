# ROCm 10 speculative acceleration

BeeLlama's optional HIP accelerator moves Qwen3.8-27B MTP catch-up or DFlash2
drafting into the dynamic `ggml-hip` backend. It is off by default and does not
replace the portable implementation. The initial validated descriptor is the
RX 7900 XTX (`gfx1100`); other GPUs continue to use `native`.

## Target-cache independence

The accelerator owns only speculative state. It never owns, decodes, converts,
or selects the target model's cache. These examples are equally valid target
configurations:

```text
--cache-type-k kvarn5 --cache-type-v kvarn5 --kv-tail-tokens 128
--cache-type-k kvarn8 --cache-type-v kvarn6 --kv-tail-tokens 1024
--cache-type-k q6_1   --cache-type-v q5_1   --kv-tail-tokens 128
--cache-type-k q8_0   --cache-type-v q3_1
```

Any target pair accepted by the normal backend remains accepted. The MTP
accelerator's separate `--spec-draft-type-k/v` pair is currently F16/F16 or
Q8_0/Q8_0. DFlash uses its own bounded F16 ring. Changing either private format
does not alter KVarN records, exact tails, prompt reuse, or target checkpoints.

## Build on Windows

The hybrid script discovers Vulkan SDK, Ninja, Windows SDK, the TheRock or
legacy ROCm layout, Clang/HIP, AMDGPU bitcode, and a physical `gfx1100` device.
It deliberately prefers the VS 2022 STL because ROCm 10's Clang 23 HIP wrappers
are not compatible with the newer VS 2026/MSVC 14.51 cmath declarations.
It also compiles and launches a one-thread HIP kernel before configuring the
project. Device enumeration alone is not sufficient: an incompatible Windows
display driver can report `gfx1100` and still reject every GPU code object.
On mixed gfx1036-iGPU/gfx1100-dGPU systems, the script automatically sets
`HIP_VISIBLE_DEVICES` for its build and test children. Current Windows ROCm can
otherwise reject a valid gfx1100 image merely because the unsupported iGPU is
also visible, even when the application explicitly selects the XTX.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-win-vulkan-rocm10-rdna3.ps1 `
  -BuildName build-vulkan-rocm10-rdna3 `
  -Target llama-server `
  -Parallel 16 `
  -Package
```

The package contains dynamic Vulkan and HIP backends and enables
`GGML_HIP_SPEC_ACCEL`. The existing Vulkan-only build script is unchanged.
Use `-RocmRoot` for a non-default installation and `-SkipDeviceCheck` only for
a cross-build host. ROCm 10's release notes list Adrenalin 26.6.4 or Windows OEM
26.10.28 as compatible Windows drivers.

## Prepare an accelerator artifact

Install the repository's GGUF Python package in an isolated environment:

```powershell
python -m venv build-spec-tools
.\build-spec-tools\Scripts\python.exe -m pip install -e .\gguf-py
```

Inspect and fingerprint the target:

```powershell
.\build-spec-tools\Scripts\python.exe .\tools\spec-accel\prepare.py inspect `
  D:\models\Qwen3.8-27B.gguf --fingerprint
```

Prepare MTP from the target's appended NextN block:

```powershell
.\build-spec-tools\Scripts\python.exe .\tools\spec-accel\prepare.py mtp `
  --target D:\models\Qwen3.8-27B.gguf `
  --output D:\models\Qwen3.8-27B-mtp-accel.gguf
```

If the target does not embed the 40,960-entry draft vocabulary, pass `--ids`
with a JSON array/object (`{"ids":[...]}`) or little-endian int32 `.bin` file.
The tool accepts supported Q4/Q5/Q6 source tensors and converts the required
artifact tensors without rewriting the target.

DFlash preparation also requires an upstream `dflash` GGUF with the trained
five-layer descriptor:

```powershell
.\build-spec-tools\Scripts\python.exe .\tools\spec-accel\prepare.py dflash `
  --target D:\models\Qwen3.8-27B.gguf `
  --draft D:\models\Qwen3.8-27B-DFlash2.gguf `
  --output D:\models\Qwen3.8-27B-dflash-accel.gguf
```

Artifacts contain a sampled target fingerprint, exact shape/type schema,
M-RoPE metadata, and vocabulary remap. Mixed-model, malformed, or unsupported
assets fail before runtime state is created.

## Run

First inspect the actual device names:

```powershell
.\build-vulkan-rocm10-rdna3\bin\llama-server.exe --list-devices
```

Set `HIP_VISIBLE_DEVICES` to the physical XTX ordinal before starting the
server on a mixed-GPU host. The filtered XTX is then reported as `ROCm0`.
Keep the target on Vulkan and pass the ROCm device only to the speculative
accelerator:

```powershell
$env:HIP_VISIBLE_DEVICES = "1" # physical XTX ordinal on this mixed-GPU host
.\build-vulkan-rocm10-rdna3\bin\llama-server.exe `
  -m D:\models\Qwen3.8-27B.gguf -ngl 999 --device Vulkan1 `
  --flash-attn on --cache-type-k kvarn5 --cache-type-v kvarn5 `
  --kv-tail-tokens 128 -b 2048 -ub 512 `
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 `
  --spec-draft-accelerator hip `
  --spec-draft-accelerator-model D:\models\Qwen3.8-27B-mtp-accel.gguf `
  --spec-draft-device ROCm0
```

The numeric suffixes above are an example only. On the test host the XTX is
`Vulkan1`; after `HIP_VISIBLE_DEVICES=1` hides the integrated GPU, the same XTX
is `ROCm0`. Use the names reported by the server process rather than assuming
either suffix.

Startup under explicit HIP selection is strict: an incompatible model, device,
private cache, or artifact is an error. After startup, a runtime fault disables
only that source for the affected slot, allowing n-gram and normal target
generation to continue.

## Focused lifecycle test

`test-spec-accel` is inert in ordinary `ctest` runs unless both model paths are
provided. On a mixed-GPU Windows host, isolate the physical XTX first; it is
then addressed as `ROCm0` inside the test process:

```powershell
$env:HIP_VISIBLE_DEVICES = "1"
$env:GGML_SPEC_ACCEL_TEST_BACKEND = "ROCm0"
$env:GGML_SPEC_ACCEL_TEST_TARGET = "D:\models\Qwen3.8-27B.gguf"
$env:GGML_SPEC_ACCEL_TEST_ARTIFACT = "D:\models\Qwen3.8-27B-mtp-accel.gguf"

$env:GGML_SPEC_ACCEL_TEST_CACHE = "f16"
.\build-vulkan-rocm10-rdna3\bin\test-spec-accel.exe
$env:GGML_SPEC_ACCEL_TEST_CACHE = "q8_0"
.\build-vulkan-rocm10-rdna3\bin\test-spec-accel.exe

# Optional when a matching DFlash artifact is already available. This adds a
# real graph-capture/selector and DFlash ring-lifecycle pass; the test never
# downloads a drafter.
$env:GGML_SPEC_ACCEL_TEST_DFLASH_ARTIFACT = "D:\models\Qwen3.8-27B-dflash-accel.gguf"
.\build-vulkan-rocm10-rdna3\bin\test-spec-accel.exe
```

The focused test runs real gfx1100 kernels and covers token catch-up, embedded
image rows with four-plane M-RoPE, record/model-position separation, drafting,
rollback, fork, scalar shift, multimodal shift rejection, transactional
checkpoints, durable state checksum/import, malformed artifact/type rejection,
slot-capacity enforcement, and three-slot isolation. With the optional DFlash
artifact it additionally exercises feature injection, quantized decode, RoPE,
attention, graph capture, selector output, rollback, fork, shift, checkpoint,
and durable ring-state restoration.

## State and fallback behavior

- Each server slot has independently addressed device state. Forks use
  device-to-device copies.
- Suffix removal and checkpoint restore change logical coverage without copying
  the complete cache into every ordinary checkpoint.
- Durable prompt state is streamed and imported transactionally; validation
  completes before live pointers or metadata are swapped.
- M-RoPE shifts move V and delta-rotate K. Unsupported multimodal shifts remain
  disabled under the server's existing multimodal policy.
- DFlash retains its live 2048-token window and checkpoint-required windows. If
  an older window is absent, DFlash waits for coverage to refill; the target is
  never replayed from zero merely to restore speculation.
- ROCm memory-pool and stream-ordered allocation are used when the runtime
  supports them, with ordinary HIP allocation as the compatibility fallback.
- Vulkan/HIP transfers are host-mediated; external-semaphore interop is not
  used.

The DFlash accelerator is opt-in and not performance-qualified without a real
matching drafter. Retain `native` unless a controlled run improves median
generation speed by at least 5% while prompt processing remains within 5% of
the native control.

Kernel portions adapted from BridgeSpec 0.1.0 retain the MIT notice in
[`licenses/BridgeSpec-MIT.txt`](../licenses/BridgeSpec-MIT.txt).
