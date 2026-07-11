"""Pure GGUF header parser.

Reads only the header (magic, version, metadata key-value section) via
seek/read — never loads a whole (potentially multi-gigabyte) model file
into memory.

Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""
import hashlib
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Optional, Union

GGUFValue = Union[int, float, bool, str, list]

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

# gguf metadata value types
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

_SCALAR_STRUCT_FORMATS = {
    _TYPE_UINT8: ("<B", 1),
    _TYPE_INT8: ("<b", 1),
    _TYPE_UINT16: ("<H", 2),
    _TYPE_INT16: ("<h", 2),
    _TYPE_UINT32: ("<I", 4),
    _TYPE_INT32: ("<i", 4),
    _TYPE_FLOAT32: ("<f", 4),
    _TYPE_BOOL: ("<?", 1),
    _TYPE_UINT64: ("<Q", 8),
    _TYPE_INT64: ("<q", 8),
    _TYPE_FLOAT64: ("<d", 8),
}

TOKEN_HASH_SAMPLE_SIZE = 256

# Guards against corrupt/malicious headers: a bogus length field could
# otherwise trigger an uncaught MemoryError (huge string) or RecursionError
# (deeply nested arrays) instead of a clean GGUFParseError.
MAX_STRING_LENGTH = 64 * 1024 * 1024  # 64 MiB — generous for any real metadata string
MAX_ARRAY_NESTING_DEPTH = 32


class GGUFParseError(Exception):
    """Raised when a file is not a valid, parseable GGUF header."""


@dataclass
class GGUFHeader:
    version: int
    architecture: Optional[str]
    tokenizer_model: Optional[str]
    vocab_size: Optional[int]
    token_hash: Optional[str]
    parameter_count: Optional[int]
    metadata_keys: list[str] = field(default_factory=list)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFParseError(
            f"unexpected end of file: wanted {n} bytes, got {len(data)} (truncated GGUF header)"
        )
    return data


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f: BinaryIO) -> str:
    length = _read_u64(f)
    if length > MAX_STRING_LENGTH:
        raise GGUFParseError(
            f"string length {length} exceeds max allowed {MAX_STRING_LENGTH} "
            "(likely a corrupt or malicious header)"
        )
    raw = _read_exact(f, length)
    return raw.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, value_type: int, _depth: int = 0) -> GGUFValue:
    if _depth > MAX_ARRAY_NESTING_DEPTH:
        raise GGUFParseError(
            f"array nesting exceeds max depth {MAX_ARRAY_NESTING_DEPTH} "
            "(likely a corrupt or malicious header)"
        )
    if value_type == _TYPE_STRING:
        return _read_string(f)
    if value_type == _TYPE_ARRAY:
        elem_type = _read_u32(f)
        length = _read_u64(f)
        return [_read_value(f, elem_type, _depth + 1) for _ in range(length)]
    if value_type in _SCALAR_STRUCT_FORMATS:
        fmt, size = _SCALAR_STRUCT_FORMATS[value_type]
        return struct.unpack(fmt, _read_exact(f, size))[0]
    raise GGUFParseError(f"unknown GGUF metadata value type: {value_type}")


def token_hash_of(tokens: list[GGUFValue]) -> str:
    """Hash the first ``TOKEN_HASH_SAMPLE_SIZE`` tokens of a vocabulary.

    Public so callers building a ``ModelInfo`` from a non-GGUF source (e.g.
    Ollama's ``/api/show``) can compute a comparable hash from the same
    ``tokenizer.ggml.tokens``-shaped list.
    """
    sample = tokens[:TOKEN_HASH_SAMPLE_SIZE]
    joined = "\x00".join(str(t) for t in sample).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def parse_header(f: BinaryIO) -> GGUFHeader:
    """Parse a GGUF header from an already-open binary file object."""
    magic = f.read(4)
    if len(magic) != 4:
        raise GGUFParseError("file too short to contain a GGUF magic number")
    if magic != GGUF_MAGIC:
        raise GGUFParseError(f"bad magic: expected {GGUF_MAGIC!r}, got {magic!r}")

    version = _read_u32(f)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFParseError(
            f"unsupported GGUF version {version}: speculect supports {SUPPORTED_VERSIONS}"
        )

    _tensor_count = _read_u64(f)  # noqa: F841 -- not needed, header-only parse
    metadata_kv_count = _read_u64(f)

    metadata: dict[str, GGUFValue] = {}
    for _ in range(metadata_kv_count):
        key = _read_string(f)
        value_type = _read_u32(f)
        metadata[key] = _read_value(f, value_type)

    architecture = metadata.get("general.architecture")
    tokenizer_model = metadata.get("tokenizer.ggml.model")
    tokens = metadata.get("tokenizer.ggml.tokens")
    parameter_count = metadata.get("general.parameter_count")

    vocab_size = None
    token_hash = None
    if isinstance(tokens, list):
        vocab_size = len(tokens)
        token_hash = token_hash_of(tokens)

    return GGUFHeader(
        version=version,
        architecture=architecture if isinstance(architecture, str) else None,
        tokenizer_model=tokenizer_model if isinstance(tokenizer_model, str) else None,
        vocab_size=vocab_size,
        token_hash=token_hash,
        parameter_count=parameter_count if isinstance(parameter_count, int) else None,
        metadata_keys=sorted(metadata.keys()),
    )


def parse_file(path: str) -> GGUFHeader:
    """Parse a GGUF header from a file on disk."""
    with open(path, "rb") as f:
        return parse_header(f)


@dataclass
class GGUFFileInfo:
    path: str
    name: str
    header: GGUFHeader


@dataclass
class ScanResult:
    files: list[GGUFFileInfo]
    skipped: list["SkippedFile"]


@dataclass
class SkippedFile:
    path: str
    reason: str


def scan_directory(directory: str) -> ScanResult:
    """Scan a directory (non-recursive) for *.gguf files and parse their headers.

    Files that fail to parse are reported in ``skipped``, not silently
    dropped — a directory scan should surface the good files *and* tell the
    caller which ones it couldn't read and why.
    """
    files: list[GGUFFileInfo] = []
    skipped: list[SkippedFile] = []
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".gguf"):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            skipped.append(SkippedFile(path=path, reason="not a regular file"))
            continue
        try:
            header = parse_file(path)
        except GGUFParseError as exc:
            skipped.append(SkippedFile(path=path, reason=str(exc)))
            continue
        files.append(GGUFFileInfo(path=path, name=name, header=header))
    return ScanResult(files=files, skipped=skipped)
