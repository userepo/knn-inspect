#!/usr/bin/env python
"""Walk a Faiss HNSW file and print every field with its offset.

This is the reference the format notes and the Rust parser are checked
against: a straight transcription of `faiss/impl/index_read.cpp` for the
layouts k-NN writes for HNSW fields, with no Faiss dependency: `IxMp` >
`IHNf` > `IxF2`/`IxFI` (float32 storage), and `IxMp` > `IHNs` > `IxSQ`
(scalar-quantized storage, such as fp16).
A file with a Lucene envelope is accepted; the footer is skipped (see
strip_envelope.py for its layout).

Usage:
    python tools/walk_faiss.py <file.faiss> [--json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

FOOTER_MAGIC = 0xC02893E8
METRICS = {0: "inner_product", 1: "l2"}
# ScalarQuantizer::QuantizerType, in declaration order.
QTYPES = {
    0: "QT_8bit",
    1: "QT_4bit",
    2: "QT_8bit_uniform",
    3: "QT_4bit_uniform",
    4: "QT_fp16",
    5: "QT_8bit_direct",
    6: "QT_6bit",
    7: "QT_bf16",
    8: "QT_8bit_direct_signed",
}
# ScalarQuantizer::RangeStat, in declaration order.
RANGESTATS = {0: "RS_minmax", 1: "RS_meanstd", 2: "RS_quantiles", 3: "RS_optim"}


class Walker:
    def __init__(self, data: bytes, base: int = 0):
        self.data = data
        self.pos = 0
        self.base = base  # offset of data[0] in the file, for reporting
        self.fields: list[dict] = []

    def _take(self, fmt: str, name: str, note: str = ""):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise EOFError(f"{name}: need {size} bytes at {self.base + self.pos}, file ends at {self.base + len(self.data)}")
        (value,) = struct.unpack_from(fmt, self.data, self.pos)
        self.fields.append({"offset": self.base + self.pos, "size": size, "field": name, "value": value, "note": note})
        self.pos += size
        return value

    def fourcc(self, name: str) -> str:
        raw = self.data[self.pos : self.pos + 4]
        if len(raw) < 4:
            raise EOFError(f"{name}: truncated fourcc at {self.base + self.pos}")
        tag = raw.decode("ascii", "replace")
        self.fields.append({"offset": self.base + self.pos, "size": 4, "field": name, "value": tag, "note": raw.hex()})
        self.pos += 4
        return tag

    def vector(self, name: str, elem_fmt: str, size_multiplier: int = 1) -> tuple[int, int]:
        """READVECTOR / READXBVECTOR: uint64 count, then count * multiplier elements."""
        count = self._take("<Q", f"{name}.len", "element count" if size_multiplier == 1 else f"x{size_multiplier} elements")
        elems = count * size_multiplier
        nbytes = elems * struct.calcsize(elem_fmt)
        if self.pos + nbytes > len(self.data):
            raise EOFError(f"{name}: {nbytes} bytes at {self.base + self.pos} run past the end")
        self.fields.append({"offset": self.base + self.pos, "size": nbytes, "field": f"{name}.data", "value": f"{elems} x {elem_fmt}", "note": ""})
        start = self.pos
        self.pos += nbytes
        return start, elems

    def index_header(self, prefix: str) -> dict:
        d = self._take("<i", f"{prefix}.d", "dimension")
        ntotal = self._take("<q", f"{prefix}.ntotal", "vector count")
        self._take("<q", f"{prefix}.dummy1", "reserved, Faiss writes 1<<20")
        self._take("<q", f"{prefix}.dummy2", "reserved, Faiss writes 1<<20")
        self._take("<?", f"{prefix}.is_trained")
        metric = self._take("<i", f"{prefix}.metric_type")
        self.fields[-1]["note"] = METRICS.get(metric, "other: metric_arg follows")
        if metric > 1:
            self._take("<f", f"{prefix}.metric_arg")
        return {"d": d, "ntotal": ntotal, "metric_type": metric}

    def walk(self) -> dict:
        out: dict = {}
        tag = self.fourcc("idmap.fourcc")
        if tag not in ("IxMp", "IxM2"):
            raise ValueError(f"expected IxMp at offset {self.base}, found {tag!r}")
        out["idmap"] = self.index_header("idmap")

        hnsw_tag = self.fourcc("hnsw.fourcc")
        if hnsw_tag not in ("IHNf", "IHNs"):
            raise ValueError(f"expected IHNf or IHNs, found {hnsw_tag!r}")
        out["hnsw"] = self.index_header("hnsw")
        self.vector("hnsw.assign_probas", "<d")
        self.vector("hnsw.cum_nneighbor_per_level", "<i")
        levels_start, nlevels_vec = self.vector("hnsw.levels", "<i")
        self.vector("hnsw.offsets", "<Q")
        self.vector("hnsw.neighbors", "<i")
        out["hnsw"]["entry_point"] = self._take("<i", "hnsw.entry_point")
        out["hnsw"]["max_level"] = self._take("<i", "hnsw.max_level")
        out["hnsw"]["ef_construction"] = self._take("<i", "hnsw.efConstruction")
        out["hnsw"]["ef_search"] = self._take("<i", "hnsw.efSearch")
        self._take("<i", "hnsw.upper_beam", "deprecated, always written as 1")
        levels = struct.unpack_from(f"<{nlevels_vec}i", self.data, levels_start)
        out["hnsw"]["level_counts"] = [sum(1 for v in levels if v > i) for i in range(out["hnsw"]["max_level"] + 1)]

        tag = self.fourcc("storage.fourcc")
        if hnsw_tag == "IHNf":
            if tag not in ("IxF2", "IxFI", "IxFl"):
                raise ValueError(f"expected flat storage, found {tag!r}")
            out["storage"] = self.index_header("storage")
            self.vector("storage.codes", "<B", size_multiplier=4)
        else:
            if tag != "IxSQ":
                raise ValueError(f"expected IxSQ under IHNs, found {tag!r}")
            out["storage"] = self.index_header("storage")
            # read_ScalarQuantizer: the enums are 4-byte ints, d and
            # code_size are size_t.
            qtype = self._take("<i", "storage.sq.qtype")
            self.fields[-1]["note"] = QTYPES.get(qtype, "unknown")
            rangestat = self._take("<i", "storage.sq.rangestat")
            self.fields[-1]["note"] = RANGESTATS.get(rangestat, "unknown")
            self._take("<f", "storage.sq.rangestat_arg")
            sq_d = self._take("<Q", "storage.sq.d")
            code_size = self._take("<Q", "storage.sq.code_size", "bytes per vector")
            self.vector("storage.sq.trained", "<f")
            # read_vector(codes): a byte vector, ntotal * code_size bytes.
            self.vector("storage.codes", "<B")
            out["storage"]["sq"] = {
                "qtype": QTYPES.get(qtype, qtype),
                "rangestat": RANGESTATS.get(rangestat, rangestat),
                "d": sq_d,
                "code_size": code_size,
            }

        self.vector("idmap.id_map", "<q")
        out["end"] = self.base + self.pos
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("file", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    data = args.file.read_bytes()
    footer = len(data) >= 16 and int.from_bytes(data[-16:-12], "big") == FOOTER_MAGIC
    payload = data[:-16] if footer else data
    w = Walker(payload)
    try:
        summary = w.walk()
    except (EOFError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        for f in w.fields[-5:]:
            print(f"  {f['offset']:>10} {f['size']:>8} {f['field']:<32} {f['value']}", file=sys.stderr)
        return 1

    trailing = len(payload) - w.pos
    summary["file_size"] = len(data)
    summary["lucene_footer"] = footer
    summary["trailing_bytes"] = trailing

    if args.json:
        print(json.dumps({"summary": summary, "fields": w.fields}, indent=2))
    else:
        print(f"{args.file}: {len(data)} bytes, lucene footer: {'yes' if footer else 'no'}")
        print(f"{'offset':>10} {'size':>8}  {'field':<32} value")
        for f in w.fields:
            note = f"  ({f['note']})" if f["note"] else ""
            print(f"{f['offset']:>10} {f['size']:>8}  {f['field']:<32} {f['value']}{note}")
        print(f"payload ends at {w.pos}; trailing bytes before footer: {trailing}")
        h = summary["hnsw"]
        print(f"summary: d={summary['idmap']['d']} n={summary['idmap']['ntotal']} metric={METRICS.get(summary['storage']['metric_type'])} "
              f"levels={h['level_counts']} entry={h['entry_point']} efC={h['ef_construction']} efS={h['ef_search']}")
    return 0 if trailing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
