#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Build compact BeeLlama speculative-accelerator GGUF artifacts.

The MTP artifact is derived entirely from user-supplied files.  It contains
only Qwen3.8's appended NextN block, a Q4_0 token embedding, a sliced Q4_0
vocabulary head, and identity metadata.  The target GGUF is never rewritten.

Parts of the tensor-selection and sliced-head preparation flow are adapted
from BridgeSpec 0.1.0.  See licenses/BridgeSpec-MIT.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    import gguf
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quants
except ImportError as exc:  # pragma: no cover - exercised by command-line users
    raise SystemExit(
        "gguf-py and its dependencies are required. Run "
        "`python -m pip install -e ./gguf-py`."
    ) from exc


ARTIFACT_VERSION = 1
MTP_BLOCK = 64
N_EMBD = 5120
N_VOCAB = 248_320
N_DRAFT_VOCAB = 40_960

ID_FIELDS = ("qwen35.nextn.draft_vocab_ids", "qwen3.nextn.draft_vocab_ids")

SUPPORTED_SOURCE_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
}

MTP_TENSORS = (
    ("token_embd.weight", (5120, 248320), "q4"),
    (f"blk.{MTP_BLOCK}.attn_k.weight", (5120, 1024), "q4"),
    (f"blk.{MTP_BLOCK}.attn_k_norm.weight", (256,), "f32"),
    (f"blk.{MTP_BLOCK}.attn_norm.weight", (5120,), "f32"),
    (f"blk.{MTP_BLOCK}.attn_output.weight", (6144, 5120), "q4"),
    (f"blk.{MTP_BLOCK}.attn_q.weight", (5120, 12288), "q4"),
    (f"blk.{MTP_BLOCK}.attn_q_norm.weight", (256,), "f32"),
    (f"blk.{MTP_BLOCK}.attn_v.weight", (5120, 1024), "q4"),
    (f"blk.{MTP_BLOCK}.ffn_down.weight", (17408, 5120), "q4"),
    (f"blk.{MTP_BLOCK}.ffn_gate.weight", (5120, 17408), "q4"),
    (f"blk.{MTP_BLOCK}.ffn_up.weight", (5120, 17408), "q4"),
    (f"blk.{MTP_BLOCK}.nextn.eh_proj.weight", (10240, 5120), "q4"),
    (f"blk.{MTP_BLOCK}.nextn.enorm.weight", (5120,), "f32"),
    (f"blk.{MTP_BLOCK}.nextn.hnorm.weight", (5120,), "f32"),
    (f"blk.{MTP_BLOCK}.nextn.shared_head_norm.weight", (5120,), "f32"),
    (f"blk.{MTP_BLOCK}.post_attention_norm.weight", (5120,), "f32"),
)


def dflash_tensors() -> tuple[tuple[str, tuple[int, ...], GGMLQuantizationType], ...]:
    result: list[tuple[str, tuple[int, ...], GGMLQuantizationType]] = [
        ("enc.output_norm.weight", (5120,), GGMLQuantizationType.F32),
        ("fc.weight", (25600, 5120), GGMLQuantizationType.Q4_K),
        ("output_norm.weight", (5120,), GGMLQuantizationType.F32),
        ("selector_hidden.weight", (5120, 256), GGMLQuantizationType.Q4_K),
        ("selector_predecessor.weight", (256, 248320), GGMLQuantizationType.Q4_K),
        ("selector_successor.weight", (256, 248320), GGMLQuantizationType.Q4_K),
    ]
    for layer in range(5):
        prefix = f"blk.{layer}."
        wide_type = GGMLQuantizationType.Q6_K if layer in (2, 4) else GGMLQuantizationType.Q4_K
        result.extend(
            [
                (prefix + "attn_conv_base", (5120, 2, 2), GGMLQuantizationType.F32),
                (prefix + "attn_conv_proj.weight", (5120, 1280), GGMLQuantizationType.Q4_K),
                (prefix + "attn_k.weight", (5120, 1024), GGMLQuantizationType.Q4_K),
                (prefix + "attn_k_norm.weight", (128,), GGMLQuantizationType.F32),
                (prefix + "attn_norm.weight", (5120,), GGMLQuantizationType.F32),
                (prefix + "attn_output.weight", (4096, 5120), GGMLQuantizationType.Q4_K),
                (prefix + "attn_q.weight", (5120, 4096), GGMLQuantizationType.Q4_K),
                (prefix + "attn_q_norm.weight", (128,), GGMLQuantizationType.F32),
                (prefix + "attn_v.weight", (5120, 1024), wide_type),
                (prefix + "ffn_conv_base", (5120, 2, 2), GGMLQuantizationType.F32),
                (prefix + "ffn_conv_proj.weight", (5120, 1280), GGMLQuantizationType.Q4_K),
                (prefix + "ffn_down.weight", (17408, 5120), wide_type),
                (prefix + "ffn_gate.weight", (5120, 17408), GGMLQuantizationType.Q4_K),
                (prefix + "ffn_norm.weight", (5120,), GGMLQuantizationType.F32),
                (prefix + "ffn_up.weight", (5120, 17408), GGMLQuantizationType.Q4_K),
            ]
        )
    return tuple(result)


DFLASH_TENSORS = dflash_tensors()

IDENTITY_TENSORS = tuple(sorted(
    ["token_embd.weight", "output.weight"] + [name for name, _, _ in MTP_TENSORS if name != "token_embd.weight"]
))


@contextmanager
def atomic_output(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as directory:
        temporary = Path(directory) / destination.name
        yield temporary
        os.replace(temporary, destination)


def open_gguf(path: Path) -> GGUFReader:
    if not path.is_file():
        raise ValueError(f"GGUF does not exist: {path}")
    if sys.byteorder != "little":
        raise ValueError("accelerator preparation currently requires a little-endian host")
    reader = GGUFReader(str(path), "r")
    if reader.byte_order != "I":
        raise ValueError("big-endian GGUF input is not supported")
    return reader


def tensor_map(reader: GGUFReader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def field_value(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    return None if field is None else field.contents()


def validate_qwen38(reader: GGUFReader) -> None:
    architecture = field_value(reader, "general.architecture")
    if architecture != "qwen35":
        raise ValueError(f"expected Qwen3.8/qwen35 target architecture, found {architecture!r}")
    checks = {
        "qwen35.embedding_length": N_EMBD,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.rope.dimension_count": 64,
    }
    for key, expected in checks.items():
        actual = field_value(reader, key)
        if actual is not None and int(actual) != expected:
            raise ValueError(f"{key}={actual!r}, expected {expected}")
    blocks = {
        int(match.group(1))
        for tensor in reader.tensors
        if (match := re.match(r"blk\.(\d+)\.nextn\.", tensor.name))
    }
    if blocks != {MTP_BLOCK}:
        raise ValueError(f"expected exactly one NextN block at {MTP_BLOCK}, found {sorted(blocks)}")


def read_ids(path: Path | None, reader: GGUFReader) -> list[int]:
    values = None
    if path is not None:
        if not path.is_file():
            raise ValueError(f"draft vocabulary file does not exist: {path}")
        if path.suffix.lower() == ".bin":
            raw = path.read_bytes()
            if len(raw) % 4:
                raise ValueError("binary draft vocabulary must be little-endian int32 data")
            values = struct.unpack(f"<{len(raw) // 4}i", raw)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload["ids"] if isinstance(payload, dict) else payload
    else:
        for key in ID_FIELDS:
            values = field_value(reader, key)
            if values is not None:
                break
    if values is None:
        raise ValueError(
            "pass --ids with the 40,960-entry Qwen3.8 draft vocabulary, or use a target "
            "GGUF that embeds qwen35.nextn.draft_vocab_ids"
        )
    ids = sorted({int(value) for value in np.asarray(values).reshape(-1).tolist()})
    if len(ids) != N_DRAFT_VOCAB:
        raise ValueError(f"expected {N_DRAFT_VOCAB:,} unique draft IDs, found {len(ids):,}")
    if ids[0] < 0 or ids[-1] >= N_VOCAB:
        raise ValueError("draft vocabulary contains an ID outside Qwen3.8's vocabulary")
    return ids


def raw_rows(tensor) -> tuple[np.ndarray, int, int]:
    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
    width = int(tensor.shape[0])
    if width % block_size:
        raise ValueError(
            f"{tensor.name}: width {width} is not divisible by {tensor.tensor_type.name}'s block size"
        )
    row_bytes = width // block_size * type_size
    raw = np.asarray(tensor.data).view(np.uint8)
    if raw.nbytes % row_bytes:
        raise ValueError(f"{tensor.name}: raw tensor bytes are not row-aligned")
    rows = raw.nbytes // row_bytes
    return raw.reshape(rows, row_bytes), rows, row_bytes


def fnv1a(value: int, data: bytes | memoryview) -> int:
    for byte in data:
        value ^= int(byte)
        value = (value * 1_099_511_628_211) & 0xFFFFFFFFFFFFFFFF
    return value


def target_fingerprint(reader: GGUFReader) -> str:
    tensors = tensor_map(reader)
    value = 1_469_598_103_934_665_603
    for name in IDENTITY_TENSORS:
        tensor = tensors.get(name)
        if tensor is None:
            raise ValueError(f"target is missing identity tensor: {name}")
        shape = [int(item) for item in tensor.shape]
        if len(shape) > 4:
            raise ValueError(f"{name}: unsupported rank {len(shape)}")
        ne = shape + [1] * (4 - len(shape))
        raw = np.asarray(tensor.data).view(np.uint8).reshape(-1)
        size = int(raw.size)
        value = fnv1a(value, name.encode("utf-8"))
        value = fnv1a(value, struct.pack("<I", int(tensor.tensor_type)))
        value = fnv1a(value, struct.pack("<4q", *ne))
        value = fnv1a(value, struct.pack("<Q", size))
        starts = (0, (size - 64) // 2 if size > 64 else 0, size - 64 if size > 64 else 0)
        for start in starts:
            count = min(64, size - start)
            value = fnv1a(value, memoryview(raw[start:start + count]))
    return f"{value:016x}"


def require_shape(tensor, expected: Sequence[int]) -> None:
    actual = tuple(int(item) for item in tensor.shape)
    if actual != tuple(expected):
        raise ValueError(f"{tensor.name}: shape {actual}, expected {tuple(expected)}")


def convert_quant(
    tensor,
    target_type: GGMLQuantizationType,
    directory: Path,
    rows_per_chunk: int,
    selected_rows: Sequence[int] | None = None,
) -> np.ndarray:
    if tensor.tensor_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"{tensor.name}: conversion from {tensor.tensor_type.name} is unsupported; "
            "use an F16/BF16 or Q2/Q3/Q4/Q5/Q6/Q8 source"
        )
    source, source_rows, _ = raw_rows(tensor)
    indices = np.arange(source_rows, dtype=np.int64) if selected_rows is None else np.asarray(selected_rows, dtype=np.int64)
    if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= source_rows:
        raise ValueError(f"{tensor.name}: selected row is outside the tensor")
    width = int(tensor.shape[0])
    target_block, target_size = gguf.GGML_QUANT_SIZES[target_type]
    if width % target_block:
        raise ValueError(f"{tensor.name}: width {width} cannot be represented as {target_type.name}")
    target_row_bytes = width // target_block * target_size
    output_path = directory / f"{len(list(directory.iterdir())):03d}.{target_type.name.lower()}"
    output = np.memmap(output_path, mode="w+", dtype=np.uint8, shape=(len(indices), target_row_bytes))
    for begin in range(0, len(indices), rows_per_chunk):
        end = min(begin + rows_per_chunk, len(indices))
        raw = np.ascontiguousarray(source[indices[begin:end]])
        if tensor.tensor_type == target_type:
            output[begin:end] = raw
            continue
        try:
            f32 = quants.dequantize(raw, tensor.tensor_type).reshape(end - begin, width)
            output[begin:end] = quants.quantize(f32, target_type)
        except NotImplementedError as exc:
            raise ValueError(
                f"{tensor.name}: gguf-py cannot convert {tensor.tensor_type.name} to {target_type.name}"
            ) from exc
    output.flush()
    return output


def convert_f32(tensor) -> np.ndarray:
    if tensor.tensor_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"{tensor.name}: cannot convert {tensor.tensor_type.name} to F32")
    try:
        return np.ascontiguousarray(
            quants.dequantize(np.asarray(tensor.data), tensor.tensor_type).reshape(*reversed(tensor.shape))
        ).astype(np.float32, copy=False)
    except NotImplementedError as exc:
        raise ValueError(f"{tensor.name}: gguf-py cannot dequantize {tensor.tensor_type.name}") from exc


def add_quant(writer: GGUFWriter, name: str, raw: np.ndarray, target_type: GGMLQuantizationType) -> None:
    writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=target_type)


def prepare_mtp(args: argparse.Namespace) -> None:
    source = args.target.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("refusing to overwrite the target GGUF")
    if destination.exists() and not args.force:
        raise ValueError(f"output exists; pass --force to replace it: {destination}")

    reader = open_gguf(source)
    validate_qwen38(reader)
    tensors = tensor_map(reader)
    ids = read_ids(args.ids, reader)
    fingerprint = target_fingerprint(reader)

    for name, shape, _ in MTP_TENSORS:
        tensor = tensors.get(name)
        if tensor is None:
            raise ValueError(f"missing required tensor: {name}")
        require_shape(tensor, shape)
    output_head = tensors.get("output.weight")
    if output_head is None:
        raise ValueError("missing required tensor: output.weight")
    require_shape(output_head, (N_EMBD, N_VOCAB))

    with atomic_output(destination) as temporary, tempfile.TemporaryDirectory(
        prefix="beellama-spec-accel-"
    ) as work:
        work_dir = Path(work)
        writer = GGUFWriter(str(temporary), "beellama-spec-accel")
        try:
            writer.add_string("general.name", f"BeeLlama Qwen3.8 MTP accelerator ({source.name})")
            writer.add_uint32("beellama.spec_accel.version", ARTIFACT_VERSION)
            writer.add_string("beellama.spec_accel.kind", "mtp")
            writer.add_string("beellama.spec_accel.target_fingerprint", fingerprint)
            writer.add_key_value(
                "beellama.spec_accel.draft_vocab_ids",
                ids,
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )
            writer.add_key_value(
                "beellama.spec_accel.mrope_sections",
                [11, 11, 10, 0],
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )

            for name, _, kind in MTP_TENSORS:
                tensor = tensors[name]
                if kind == "q4":
                    print(f"converting {name}: {tensor.tensor_type.name} -> Q4_0", flush=True)
                    add_quant(
                        writer,
                        name,
                        convert_quant(tensor, GGMLQuantizationType.Q4_0, work_dir, args.rows_per_chunk),
                        GGMLQuantizationType.Q4_0,
                    )
                else:
                    writer.add_tensor(name, convert_f32(tensor))

            head_name = f"blk.{MTP_BLOCK}.nextn.shared_head_head.weight"
            print(
                f"slicing {len(ids):,}/{N_VOCAB:,} output rows: "
                f"{output_head.tensor_type.name} -> Q4_0",
                flush=True,
            )
            add_quant(
                writer,
                head_name,
                convert_quant(
                    output_head,
                    GGMLQuantizationType.Q4_0,
                    work_dir,
                    args.rows_per_chunk,
                    selected_rows=ids,
                ),
                GGMLQuantizationType.Q4_0,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        finally:
            writer.close()

    print(f"wrote {destination} (target fingerprint {fingerprint})")


def require_int(reader: GGUFReader, key: str, expected: int) -> None:
    value = field_value(reader, key)
    if value is None or int(value) != expected:
        raise ValueError(f"{key}={value!r}, expected {expected}")


def validate_dflash(reader: GGUFReader) -> tuple[list[int], list[int], float, int]:
    if field_value(reader, "general.architecture") != "dflash":
        raise ValueError("the DFlash accelerator requires an upstream dflash GGUF")
    for key, expected in {
        "dflash.embedding_length": 5120,
        "dflash.block_count": 5,
        "dflash.feed_forward_length": 17408,
        "dflash.attention.head_count": 32,
        "dflash.attention.head_count_kv": 8,
        "dflash.attention.key_length": 128,
        "dflash.attention.value_length": 128,
        "dflash.block_size": 8,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
    }.items():
        require_int(reader, key, expected)

    target_layers = [int(value) for value in np.asarray(
        field_value(reader, "dflash.target_layers") or []
    ).reshape(-1).tolist()]
    if len(target_layers) != 5 or len(set(target_layers)) != 5 or min(target_layers) < 0:
        raise ValueError(f"dflash.target_layers must contain five unique non-negative layers, found {target_layers}")

    window = field_value(reader, "dflash.attention.sliding_window")
    if window is None or int(window) != 2048:
        raise ValueError(f"DFlash HIP currently requires a 2048-token sliding window, found {window!r}")
    pattern = field_value(reader, "dflash.attention.sliding_window_pattern")
    if pattern is not None:
        values = [bool(value) for value in np.asarray(pattern).reshape(-1).tolist()]
        if len(values) != 5 or not all(values):
            raise ValueError("DFlash HIP only accepts models proven to use sliding-window attention in every layer")

    sections_value = field_value(reader, "dflash.rope.dimension_sections")
    sections = [64, 0, 0, 0] if sections_value is None else [
        int(value) for value in np.asarray(sections_value).reshape(-1).tolist()
    ]
    if len(sections) != 4 or any(value < 0 for value in sections) or sum(sections) != 64:
        raise ValueError(f"unsupported DFlash M-RoPE sections: {sections}")
    freq_base_value = field_value(reader, "dflash.rope.freq_base")
    freq_base = 10_000_000.0 if freq_base_value is None else float(freq_base_value)
    if not np.isfinite(freq_base) or freq_base <= 0:
        raise ValueError(f"invalid DFlash RoPE frequency base: {freq_base!r}")
    mask_token_value = field_value(reader, "tokenizer.ggml.mask_token_id")
    if mask_token_value is None:
        raise ValueError("DFlash GGUF is missing tokenizer.ggml.mask_token_id")
    mask_token_id = int(mask_token_value)
    if mask_token_id < 0 or mask_token_id >= N_VOCAB:
        raise ValueError(f"invalid DFlash mask token ID: {mask_token_id}")
    return target_layers, sections, freq_base, mask_token_id


def prepare_dflash(args: argparse.Namespace) -> None:
    target_path = args.target.resolve()
    draft_path = args.draft.resolve()
    destination = args.output.resolve()
    if destination in (target_path, draft_path):
        raise ValueError("refusing to overwrite a source GGUF")
    if destination.exists() and not args.force:
        raise ValueError(f"output exists; pass --force to replace it: {destination}")

    target = open_gguf(target_path)
    validate_qwen38(target)
    draft = open_gguf(draft_path)
    target_layers, sections, freq_base, mask_token_id = validate_dflash(draft)
    ids = read_ids(args.ids, target)
    fingerprint = target_fingerprint(target)
    target_tensors = tensor_map(target)
    draft_tensors_map = tensor_map(draft)

    for name, shape, _ in DFLASH_TENSORS:
        tensor = draft_tensors_map.get(name)
        if tensor is None:
            raise ValueError(f"DFlash model is missing required tensor: {name}")
        require_shape(tensor, shape)
    embedding = target_tensors.get("token_embd.weight")
    head = target_tensors.get("output.weight")
    if embedding is None or head is None:
        raise ValueError("target is missing token_embd.weight or output.weight")
    require_shape(embedding, (N_EMBD, N_VOCAB))
    require_shape(head, (N_EMBD, N_VOCAB))

    with atomic_output(destination) as temporary, tempfile.TemporaryDirectory(
        prefix="beellama-spec-accel-"
    ) as work:
        work_dir = Path(work)
        writer = GGUFWriter(str(temporary), "beellama-spec-accel")
        try:
            writer.add_string("general.name", f"BeeLlama Qwen3.8 DFlash2 accelerator ({draft_path.name})")
            writer.add_uint32("beellama.spec_accel.version", ARTIFACT_VERSION)
            writer.add_string("beellama.spec_accel.kind", "dflash")
            writer.add_string("beellama.spec_accel.target_fingerprint", fingerprint)
            writer.add_key_value(
                "beellama.spec_accel.draft_vocab_ids",
                ids,
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )
            writer.add_key_value(
                "beellama.spec_accel.target_layers",
                target_layers,
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )
            writer.add_key_value(
                "beellama.spec_accel.mrope_sections",
                sections,
                GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32,
            )
            writer.add_float32("beellama.spec_accel.rope_freq_base", freq_base)
            writer.add_uint32("beellama.spec_accel.sliding_window", 2048)
            writer.add_int32("beellama.spec_accel.mask_token_id", mask_token_id)

            for name, _, target_type in DFLASH_TENSORS:
                tensor = draft_tensors_map[name]
                if target_type == GGMLQuantizationType.F32:
                    writer.add_tensor(name, convert_f32(tensor))
                    continue
                print(f"converting {name}: {tensor.tensor_type.name} -> {target_type.name}", flush=True)
                add_quant(
                    writer,
                    name,
                    convert_quant(tensor, target_type, work_dir, args.rows_per_chunk),
                    target_type,
                )

            print(f"converting target embedding: {embedding.tensor_type.name} -> Q4_0", flush=True)
            add_quant(
                writer,
                "target.token_embd.weight",
                convert_quant(embedding, GGMLQuantizationType.Q4_0, work_dir, args.rows_per_chunk),
                GGMLQuantizationType.Q4_0,
            )
            print(f"slicing target head: {head.tensor_type.name} -> Q6_K", flush=True)
            add_quant(
                writer,
                "target.output.weight",
                convert_quant(
                    head,
                    GGMLQuantizationType.Q6_K,
                    work_dir,
                    args.rows_per_chunk,
                    selected_rows=ids,
                ),
                GGMLQuantizationType.Q6_K,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        finally:
            writer.close()

    print(f"wrote {destination} (target fingerprint {fingerprint})")


def inspect(args: argparse.Namespace) -> None:
    reader = open_gguf(args.model)
    print(f"model={args.model}")
    print(f"architecture={field_value(reader, 'general.architecture')}")
    print(f"tensors={len(reader.tensors)}")
    for name in ("token_embd.weight", "output.weight"):
        tensor = tensor_map(reader).get(name)
        if tensor is not None:
            print(f"{name}: {tensor.tensor_type.name} {[int(item) for item in tensor.shape]}")
    nextn_blocks = sorted({
        int(match.group(1))
        for tensor in reader.tensors
        if (match := re.match(r"blk\.(\d+)\.nextn\.", tensor.name))
    })
    print(f"nextn_tensors={sum('.nextn.' in tensor.name for tensor in reader.tensors)}")
    print(f"nextn_blocks={nextn_blocks}")
    for key in ID_FIELDS:
        values = field_value(reader, key)
        if values is not None:
            print(f"{key}={np.asarray(values).size} entries")
    if args.fingerprint:
        validate_qwen38(reader)
        print(f"target_fingerprint={target_fingerprint(reader)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    view = commands.add_parser("inspect", help="inspect relevant GGUF metadata without converting")
    view.add_argument("model", type=Path)
    view.add_argument("--fingerprint", action="store_true")
    view.set_defaults(run=inspect)

    mtp = commands.add_parser("mtp", help="prepare a compact Qwen3.8 MTP accelerator GGUF")
    mtp.add_argument("--target", type=Path, required=True)
    mtp.add_argument("--ids", type=Path)
    mtp.add_argument("--output", type=Path, required=True)
    mtp.add_argument("--rows-per-chunk", type=int, default=128)
    mtp.add_argument("--force", action="store_true")
    mtp.set_defaults(run=prepare_mtp)

    dflash = commands.add_parser("dflash", help="prepare a compact Qwen3.8 DFlash2 accelerator GGUF")
    dflash.add_argument("--target", type=Path, required=True)
    dflash.add_argument("--draft", type=Path, required=True)
    dflash.add_argument("--ids", type=Path)
    dflash.add_argument("--output", type=Path, required=True)
    dflash.add_argument("--rows-per-chunk", type=int, default=128)
    dflash.add_argument("--force", action="store_true")
    dflash.set_defaults(run=prepare_dflash)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "rows_per_chunk", 1) <= 0:
        parser.error("--rows-per-chunk must be positive")
    try:
        args.run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
