import io
import struct

import pytest
from conftest import gguf_string, string_array

from speculect.gguf import (
    GGUFParseError,
    parse_file,
    parse_header,
    scan_directory,
    token_hash_of,
)


def test_parse_valid_header_v3(gguf_builder, model_metadata):
    data = gguf_builder(version=3, metadata=model_metadata())
    header = parse_header(io.BytesIO(data))
    assert header.version == 3
    assert header.architecture == "llama"
    assert header.tokenizer_model == "llama"
    assert header.vocab_size == 5
    assert header.token_hash is not None
    assert header.parameter_count == 1_000_000_000


def test_parse_valid_header_v2(gguf_builder, model_metadata):
    data = gguf_builder(version=2, metadata=model_metadata())
    header = parse_header(io.BytesIO(data))
    assert header.version == 2
    assert header.architecture == "llama"


def test_bad_magic(gguf_builder, model_metadata):
    data = gguf_builder(magic=b"NOPE", metadata=model_metadata())
    with pytest.raises(GGUFParseError, match="bad magic"):
        parse_header(io.BytesIO(data))


def test_truncated_file(gguf_builder, model_metadata):
    data = gguf_builder(metadata=model_metadata())
    truncated = data[: len(data) // 2]
    with pytest.raises(GGUFParseError, match="unexpected end of file"):
        parse_header(io.BytesIO(truncated))


def test_truncated_at_magic_only():
    with pytest.raises(GGUFParseError):
        parse_header(io.BytesIO(b"GG"))


def test_unsupported_version(gguf_builder, model_metadata):
    data = gguf_builder(version=99, metadata=model_metadata())
    with pytest.raises(GGUFParseError, match="unsupported GGUF version"):
        parse_header(io.BytesIO(data))


def test_missing_tokenizer_metadata(gguf_builder):
    metadata = {"general.architecture": gguf_string("llama")}
    data = gguf_builder(metadata=metadata)
    header = parse_header(io.BytesIO(data))
    assert header.architecture == "llama"
    assert header.tokenizer_model is None
    assert header.vocab_size is None
    assert header.token_hash is None


def test_empty_metadata(gguf_builder):
    data = gguf_builder(metadata={})
    header = parse_header(io.BytesIO(data))
    assert header.architecture is None
    assert header.metadata_keys == []


def test_token_hash_deterministic():
    tokens = ["a", "b", "c"]
    assert token_hash_of(tokens) == token_hash_of(list(tokens))


def test_token_hash_differs_for_different_tokens():
    assert token_hash_of(["a", "b"]) != token_hash_of(["a", "c"])


def test_token_hash_only_samples_first_256():
    base = [f"tok{i}" for i in range(256)]
    a = base + ["only-in-a"]
    b = base + ["only-in-b"]
    assert token_hash_of(a) == token_hash_of(b)


def test_parse_file_reads_from_disk(tmp_path, gguf_builder, model_metadata):
    data = gguf_builder(metadata=model_metadata())
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    header = parse_file(str(path))
    assert header.architecture == "llama"


def test_parse_file_does_not_load_whole_file(tmp_path, gguf_builder, model_metadata):
    # header is tiny, but the file itself is large (simulated tensor payload) —
    # parse_file must return quickly and correctly regardless of trailing size.
    data = gguf_builder(metadata=model_metadata())
    path = tmp_path / "big.gguf"
    with open(path, "wb") as f:
        f.write(data)
        f.write(b"\x00" * (5 * 1024 * 1024))  # 5MB of fake tensor data
    header = parse_file(str(path))
    assert header.architecture == "llama"


def test_scan_directory_finds_valid_files(tmp_path, gguf_builder, model_metadata):
    (tmp_path / "a.gguf").write_bytes(gguf_builder(metadata=model_metadata(architecture="llama")))
    (tmp_path / "b.gguf").write_bytes(gguf_builder(metadata=model_metadata(architecture="gemma")))
    (tmp_path / "ignore.txt").write_text("not a gguf file")
    result = scan_directory(str(tmp_path))
    names = sorted(f.name for f in result.files)
    assert names == ["a.gguf", "b.gguf"]
    assert result.skipped == []


def test_scan_directory_reports_corrupt_files_not_silently(tmp_path, gguf_builder, model_metadata):
    (tmp_path / "good.gguf").write_bytes(gguf_builder(metadata=model_metadata()))
    (tmp_path / "corrupt.gguf").write_bytes(b"NOTGGUF garbage bytes here")
    result = scan_directory(str(tmp_path))
    assert len(result.files) == 1
    assert result.files[0].name == "good.gguf"
    assert len(result.skipped) == 1
    assert result.skipped[0].path.endswith("corrupt.gguf")
    assert "bad magic" in result.skipped[0].reason


def test_scan_directory_empty(tmp_path):
    result = scan_directory(str(tmp_path))
    assert result.files == []
    assert result.skipped == []


def test_unicode_model_name_in_metadata(gguf_builder):
    metadata = {
        "general.architecture": gguf_string("llama"),
        "general.name": gguf_string("模型-🦙-café"),
        "tokenizer.ggml.model": gguf_string("llama"),
        "tokenizer.ggml.tokens": string_array(["<unk>", "日本語", "😀"]),
    }
    data = gguf_builder(metadata=metadata)
    header = parse_header(io.BytesIO(data))
    assert header.architecture == "llama"
    assert header.vocab_size == 3
    assert header.token_hash is not None


def test_array_of_arrays_not_needed_but_scalar_types_all_readable(gguf_builder):
    # exercise every scalar struct format at least once
    from speculect.gguf import (
        _TYPE_BOOL,
        _TYPE_FLOAT32,
        _TYPE_FLOAT64,
        _TYPE_INT8,
        _TYPE_INT16,
        _TYPE_INT32,
        _TYPE_INT64,
        _TYPE_UINT8,
        _TYPE_UINT16,
        _TYPE_UINT32,
        _TYPE_UINT64,
    )

    metadata = {
        "a.u8": (_TYPE_UINT8, 200),
        "a.i8": (_TYPE_INT8, -5),
        "a.u16": (_TYPE_UINT16, 40000),
        "a.i16": (_TYPE_INT16, -1000),
        "a.u32": (_TYPE_UINT32, 4_000_000_000),
        "a.i32": (_TYPE_INT32, -2_000_000_000),
        "a.f32": (_TYPE_FLOAT32, 1.5),
        "a.bool": (_TYPE_BOOL, True),
        "a.u64": (_TYPE_UINT64, 10_000_000_000),
        "a.i64": (_TYPE_INT64, -10_000_000_000),
        "a.f64": (_TYPE_FLOAT64, 3.14159),
        "general.architecture": gguf_string("llama"),
    }
    data = gguf_builder(metadata=metadata)
    header = parse_header(io.BytesIO(data))
    assert header.architecture == "llama"
    assert len(header.metadata_keys) == 12


def _malformed_bytes(kv_bytes, kv_count=1):
    """Hand-build a GGUF header with exactly `kv_bytes` as the metadata section body."""
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))
    buf.write(struct.pack("<Q", 0))
    buf.write(struct.pack("<Q", kv_count))
    buf.write(kv_bytes)
    return buf.getvalue()


def test_huge_declared_string_length_raises_parse_error_not_memory_error():
    key = b"s"
    kv = struct.pack("<Q", len(key)) + key
    kv += struct.pack("<I", 8)  # STRING type
    kv += struct.pack("<Q", 2**40)  # huge declared length, no data follows
    with pytest.raises(GGUFParseError, match="exceeds max"):
        parse_header(io.BytesIO(_malformed_bytes(kv)))


def test_deeply_nested_arrays_raise_parse_error_not_recursion_error():
    key = b"nested"
    kv = struct.pack("<Q", len(key)) + key
    kv += struct.pack("<I", 9)  # outer value_type = ARRAY
    wrap_count = 40  # comfortably past MAX_ARRAY_NESTING_DEPTH
    for _ in range(wrap_count):
        kv += struct.pack("<I", 9)  # elem_type = ARRAY
        kv += struct.pack("<Q", 1)  # one nested element
    kv += struct.pack("<I", 0)  # innermost elem_type = UINT8
    kv += struct.pack("<Q", 1)
    kv += struct.pack("<B", 42)
    with pytest.raises(GGUFParseError, match="nesting exceeds max depth"):
        parse_header(io.BytesIO(_malformed_bytes(kv)))


def test_scan_directory_reports_non_regular_file_not_silently(tmp_path):
    (tmp_path / "dir.gguf").mkdir()
    result = scan_directory(str(tmp_path))
    assert result.files == []
    assert len(result.skipped) == 1
    assert result.skipped[0].reason == "not a regular file"


def test_non_string_array_metadata_does_not_crash(gguf_builder):
    from speculect.gguf import _TYPE_ARRAY, _TYPE_INT32

    metadata = {
        "general.architecture": gguf_string("llama"),
        "some.int.array": (_TYPE_ARRAY, (_TYPE_INT32, [1, 2, 3])),
    }
    data = gguf_builder(metadata=metadata)
    header = parse_header(io.BytesIO(data))
    assert header.architecture == "llama"
    assert "some.int.array" in header.metadata_keys
