#!/usr/bin/env python
"""Walk a Faiss HNSW file and print every field with its offset.

This is the reference the format notes and the Rust parser are checked
against: a straight transcription of `faiss/impl/index_read.cpp`, plus
the `null` storage k-NN's patched writer leaves, for the layouts k-NN
writes for HNSW fields, with no Faiss dependency: `IxMp` >
`IHNf` > `IxF2`/`IxFI` (float32 storage), `IxMp` > `IHNs` > `IxSQ`
(scalar-quantized storage, such as fp16), `IBMp` > `IBHf` > `IBxF`
(binary storage, packed bits), and `IBMp` > `IBHf` > `null` (a graph
without its vectors, which k-NN writes for an `on_disk` field and keeps
the quantized vectors in Lucene's files); and the CAGRA graphs k-NN's
remote index build writes in place of the HNSW one: `IHc2` (or Faiss's
older `IHNc`) over any of the float storages or `null`, and `IBHc` over
`IBxF`, each with its CAGRA fields between the header and the graph.
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

    def binary_header(self, prefix: str) -> dict:
        """read_index_binary_header: no reserved fields, no metric argument."""
        d = self._take("<i", f"{prefix}.d", "dimension: bits over IBxF, components over null")
        code_size = self._take("<i", f"{prefix}.code_size", "bytes per vector")
        ntotal = self._take("<q", f"{prefix}.ntotal", "vector count")
        self._take("<?", f"{prefix}.is_trained")
        metric = self._take("<i", f"{prefix}.metric_type", "over IBxF unused (Hamming distance); over null, the build metric")
        return {"d": d, "code_size": code_size, "ntotal": ntotal, "metric_type": metric}

    def walk(self) -> dict:
        out: dict = {}
        tag = self.fourcc("idmap.fourcc")
        if tag == "IBMp":
            return self.walk_binary(out)
        if tag not in ("IxMp", "IxM2"):
            raise ValueError(f"expected IxMp or IBMp at offset {self.base}, found {tag!r}")
        out["idmap"] = self.index_header("idmap")

        hnsw_tag = self.fourcc("hnsw.fourcc")
        if hnsw_tag not in ("IHNf", "IHNs", "IHc2", "IHNc"):
            raise ValueError(f"expected IHNf, IHNs, IHc2, or IHNc, found {hnsw_tag!r}")
        out["hnsw"] = self.index_header("hnsw")
        out["graph"] = "hnsw"
        if hnsw_tag in ("IHc2", "IHNc"):
            out["graph"] = "cagra"
            out["cagra"] = self.cagra_fields(with_numeric_type=hnsw_tag == "IHc2")
        self.hnsw_graph(out)
        self.storage(out, hnsw_tag)
        self.vector("idmap.id_map", "<q")
        out["end"] = self.base + self.pos
        return out

    def walk_binary(self, out: dict) -> dict:
        out["idmap"] = self.binary_header("idmap")
        hnsw_tag = self.fourcc("hnsw.fourcc")
        if hnsw_tag not in ("IBHf", "IBHc"):
            raise ValueError(f"expected IBHf or IBHc under IBMp, found {hnsw_tag!r}")
        out["hnsw"] = self.binary_header("hnsw")
        out["graph"] = "hnsw"
        if hnsw_tag == "IBHc":
            out["graph"] = "cagra"
            out["cagra"] = self.cagra_fields(with_numeric_type=False)
        self.hnsw_graph(out)
        tag = self.fourcc("storage.fourcc")
        if tag == "null" and hnsw_tag == "IBHf":
            # k-NN's patch 0008: the storage left out, the tag alone.
            self.fields[-1]["note"] = "no storage: the vectors are in Lucene's files"
            out["storage"] = None
        elif tag == "IBxF":
            out["storage"] = self.binary_header("storage")
            # read_vector(xb): a byte vector, ntotal * code_size bytes.
            self.vector("storage.codes", "<B")
        else:
            expected = "IBxF or null under IBHf" if hnsw_tag == "IBHf" else "IBxF under IBHc"
            raise ValueError(f"expected {expected}, found {tag!r}")
        self.vector("idmap.id_map", "<q")
        out["end"] = self.base + self.pos
        return out

    def cagra_fields(self, with_numeric_type: bool) -> dict:
        """The fields `IndexHNSWCagra` and `IndexBinaryHNSWCagra` store after the header.

        Two bools and an int32; `IHc2` adds the storage's numeric type as
        an int32, which k-NN reads and discards.
        """
        keep = self._take("<?", "cagra.keep_max_size_level0", "level 0 lists kept at full size while building")
        base_only = self._take("<?", "cagra.base_level_only", "searched on level 0 alone, from random entry points")
        entrypoints = self._take("<i", "cagra.num_base_level_search_entrypoints", "random entry points drawn per search")
        fields = {"keep_max_size_level0": keep, "base_level_only": base_only, "num_base_level_search_entrypoints": entrypoints}
        if with_numeric_type:
            numeric = self._take("<i", "cagra.numeric_type", "0 Float32, 1 Float16, 2 UInt8, 3 Int8; stored, never read")
            fields["numeric_type"] = numeric
        return fields

    def hnsw_graph(self, out: dict) -> None:
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
        # The levels a search reads: max_level + 1, or level 0 alone for a
        # graph of nodes whose max_level is -1, as a CAGRA graph's is.
        searched = max(out["hnsw"]["max_level"] + 1, 1 if nlevels_vec else 0)
        out["hnsw"]["level_counts"] = [sum(1 for v in levels if v > i) for i in range(searched)]

    def storage(self, out: dict, hnsw_tag: str) -> None:
        tag = self.fourcc("storage.fourcc")
        cagra = hnsw_tag in ("IHc2", "IHNc")
        if cagra and tag == "null":
            # k-NN's remote build leaves the storage out for a float field
            # whose sq encoder keeps 1, 2, or 4 bits.
            self.fields[-1]["note"] = "no storage: the vectors are in Lucene's files"
            out["storage"] = None
        elif hnsw_tag == "IHNf" or (cagra and tag != "IxSQ"):
            if tag not in ("IxF2", "IxFI", "IxFl"):
                expected = "IxF2, IxFI, IxSQ, or null under a CAGRA graph" if cagra else "flat storage"
                raise ValueError(f"expected {expected}, found {tag!r}")
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
        storage = summary["storage"]
        if storage is None:
            # A graph without its vectors records the metric it was built
            # for; k-NN reads the ID map's.
            metric = METRICS.get(summary["idmap"]["metric_type"])
        else:
            metric = "hamming" if "code_size" in storage else METRICS.get(storage["metric_type"])
        cagra = summary.get("cagra")
        graph = "" if cagra is None else (
            f" graph=cagra(base_level_only={int(cagra['base_level_only'])}, "
            f"entrypoints={cagra['num_base_level_search_entrypoints']})"
        )
        print(f"summary: d={summary['idmap']['d']} n={summary['idmap']['ntotal']} metric={metric} "
              f"levels={h['level_counts']} entry={h['entry_point']} efC={h['ef_construction']} efS={h['ef_search']}{graph}")
    return 0 if trailing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
