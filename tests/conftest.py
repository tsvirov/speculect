"""Programmatic GGUF byte fixtures — no real model files, ever.

``build_gguf_bytes`` writes valid GGUF header bytes by hand (magic, version,
tensor count, metadata KV section), so tests never depend on downloading or
shipping a real (multi-gigabyte) model.
"""
import io
import struct

import pytest

_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<?",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}


def _write_string(buf, s):
    raw = s.encode("utf-8")
    buf.write(struct.pack("<Q", len(raw)))
    buf.write(raw)


def _write_value(buf, value_type, value):
    if value_type == _TYPE_STRING:
        _write_string(buf, value)
        return
    if value_type == _TYPE_ARRAY:
        elem_type, items = value
        buf.write(struct.pack("<I", elem_type))
        buf.write(struct.pack("<Q", len(items)))
        for item in items:
            _write_value(buf, elem_type, item)
        return
    fmt = _SCALAR_FORMATS[value_type]
    buf.write(struct.pack(fmt, value))


def build_gguf_bytes(version=3, metadata=None, magic=b"GGUF", tensor_count=0):
    """Build valid GGUF header bytes.

    ``metadata`` maps key -> (value_type, value), where value_type is one of
    the module-level ``_TYPE_*`` constants and, for ``_TYPE_ARRAY``, value is
    itself ``(elem_type, list_of_items)``.
    """
    metadata = metadata or {}
    buf = io.BytesIO()
    buf.write(magic)
    buf.write(struct.pack("<I", version))
    buf.write(struct.pack("<Q", tensor_count))
    buf.write(struct.pack("<Q", len(metadata)))
    for key, (value_type, value) in metadata.items():
        _write_string(buf, key)
        buf.write(struct.pack("<I", value_type))
        _write_value(buf, value_type, value)
    return buf.getvalue()


def string_array(items):
    return (_TYPE_ARRAY, (_TYPE_STRING, items))


def gguf_string(value):
    return (_TYPE_STRING, value)


def gguf_uint32(value):
    return (_TYPE_UINT32, value)


def gguf_uint64(value):
    return (_TYPE_UINT64, value)


def valid_model_metadata(architecture="llama", tokenizer_model="llama", vocab=None,
                          parameter_count=1_000_000_000):
    if vocab is None:
        vocab = ["<unk>", "<s>", "</s>", "hello", "world"]
    return {
        "general.architecture": gguf_string(architecture),
        "general.parameter_count": gguf_uint64(parameter_count),
        "tokenizer.ggml.model": gguf_string(tokenizer_model),
        "tokenizer.ggml.tokens": string_array(vocab),
    }


@pytest.fixture
def gguf_builder():
    return build_gguf_bytes


@pytest.fixture
def model_metadata():
    return valid_model_metadata
