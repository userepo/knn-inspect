#!/usr/bin/env python
"""Report and strip the Lucene codec envelope around a k-NN native file.

The OpenSearch k-NN plugin streams the Faiss bytes through a Lucene
`IndexOutput` and ends the file with `CodecUtil.writeFooter`: a 16-byte,
big-endian footer of magic `0xC02893E8`, algorithm id 0, and the CRC32 of
every preceding byte stored as an int64. No header is written, but one
is recognized and removed if present (`CodecUtil.writeIndexHeader`).

With `--out`, the bare Faiss payload is written so `faiss.read_index` can
load it. Without it, the envelope is only reported and verified.

Usage:
    python tools/strip_envelope.py <file.faiss> [--out payload.faiss] [--json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

CODEC_MAGIC = 0x3FD76C17
FOOTER_MAGIC = 0xC02893E8  # ~CODEC_MAGIC as uint32
FOOTER_LEN = 16
ID_LENGTH = 16  # StringHelper.ID_LENGTH


def read_vint(data: bytes, pos: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def parse_header(data: bytes) -> dict | None:
    if len(data) < 4 or int.from_bytes(data[:4], "big") != CODEC_MAGIC:
        return None
    name_len, pos = read_vint(data, 4)
    codec = data[pos : pos + name_len].decode("utf-8")
    pos += name_len
    version = int.from_bytes(data[pos : pos + 4], "big", signed=True)
    pos += 4
    segment_id = data[pos : pos + ID_LENGTH].hex()
    pos += ID_LENGTH
    suffix_len = data[pos]
    pos += 1
    suffix = data[pos : pos + suffix_len].decode("utf-8")
    pos += suffix_len
    return {"length": pos, "codec": codec, "version": version, "segment_id": segment_id, "suffix": suffix}


def parse_footer(data: bytes) -> dict | None:
    if len(data) < FOOTER_LEN:
        return None
    tail = data[-FOOTER_LEN:]
    magic = int.from_bytes(tail[0:4], "big")
    if magic != FOOTER_MAGIC:
        return None
    algorithm = int.from_bytes(tail[4:8], "big", signed=True)
    stored = int.from_bytes(tail[8:16], "big", signed=True)
    # The CRC covers everything before the checksum itself, the footer's
    # magic and algorithm id included.
    actual = zlib.crc32(data[:-8])
    return {
        "offset": len(data) - FOOTER_LEN,
        "algorithm": algorithm,
        "stored_crc32": stored,
        "actual_crc32": actual,
        "crc32_ok": stored == actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("file", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, help="write the bare Faiss payload here")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    data = args.file.read_bytes()
    header = parse_header(data)
    footer = parse_footer(data)
    start = header["length"] if header else 0
    end = footer["offset"] if footer else len(data)
    payload = data[start:end]
    report = {
        "file": str(args.file),
        "size": len(data),
        "header": header,
        "footer": footer,
        "payload": {"offset": start, "length": len(payload), "fourcc": payload[:4].decode("ascii", "replace")},
    }

    if args.out:
        args.out.write_bytes(payload)
        report["out"] = str(args.out)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{args.file}: {len(data)} bytes")
        print(f"  header : {'none' if header is None else header}")
        if footer is None:
            print("  footer : none")
        else:
            status = "ok" if footer["crc32_ok"] else f"MISMATCH actual=0x{footer['actual_crc32']:08x}"
            print(f"  footer : at {footer['offset']}, algorithm {footer['algorithm']}, crc32 0x{footer['stored_crc32']:08x} {status}")
        print(f"  payload: offset {start}, {len(payload)} bytes, fourcc {report['payload']['fourcc']!r}")
        if args.out:
            print(f"  written: {args.out}")

    return 0 if footer is None or footer["crc32_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
