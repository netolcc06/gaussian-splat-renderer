#!/usr/bin/env python3
"""Inspect a 3D Gaussian Splatting .ply file (INRIA graphdeco format).

Prints the header (format, vertex count, property list), the inferred SH degree,
and - unless --header-only is given - statistics over the payload:
bounding box, opacity range after sigmoid, scale range after exp, quaternion
norms, DC color range, NaN/Inf counts.

Pure standard library (no numpy) so it runs with `python -I`.
See docs/notes/ply-format.md for the format specification.

Usage:
    python -I tools/ply_info.py <file.ply> [more.ply ...] [--header-only] [--json]
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import struct
import sys

# PLY scalar type name -> (struct code, size in bytes). Both old and new names.
PLY_TYPES = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}

SH_C0 = 0.28209479177387814
MAX_HEADER_BYTES = 1 << 20  # sanity bound; real headers are ~2 KB


class PlyError(Exception):
    pass


def read_header(f):
    """Parse the ASCII header. Returns dict with format, elements, header_size."""
    first = f.readline()
    if first.rstrip(b"\r\n") != b"ply":
        raise PlyError("not a PLY file (missing 'ply' magic)")
    fmt = None
    elements = []  # list of {name, count, props: [(type, name)] , list_props}
    comments = []
    total = len(first)
    while True:
        line = f.readline()
        if not line:
            raise PlyError("unexpected EOF in header")
        total += len(line)
        if total > MAX_HEADER_BYTES:
            raise PlyError("header too large")
        tok = line.decode("ascii", "replace").strip().split()
        if not tok:
            continue
        kw = tok[0]
        if kw == "end_header":
            break
        if kw == "format":
            fmt = tok[1]
        elif kw in ("comment", "obj_info"):
            comments.append(" ".join(tok[1:]))
        elif kw == "element":
            elements.append({"name": tok[1], "count": int(tok[2]), "props": [], "has_list": False})
        elif kw == "property":
            if not elements:
                raise PlyError("property before element")
            if tok[1] == "list":
                elements[-1]["has_list"] = True
                elements[-1]["props"].append(("list", tok[-1]))
            else:
                if tok[1] not in PLY_TYPES:
                    raise PlyError(f"unknown property type {tok[1]}")
                elements[-1]["props"].append((tok[1], tok[2]))
        else:
            raise PlyError(f"unknown header keyword {kw!r}")
    if fmt is None:
        raise PlyError("missing format line")
    return {"format": fmt, "elements": elements, "comments": comments, "header_size": total}


def sh_degree_from_rest(n_rest: int):
    """f_rest count = 3 * ((deg+1)^2 - 1): 0 -> 0, 9 -> 1, 24 -> 2, 45 -> 3."""
    for deg in range(0, 5):
        if n_rest == 3 * ((deg + 1) ** 2 - 1):
            return deg
    return None


def classify(props):
    names = [n for _, n in props]
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [r for r in required if r not in names]
    # f_rest must be contiguous 0..n-1 for the standard layout
    rest_ok = all(f"f_rest_{i}" in names for i in range(n_rest))
    normals = [n for n in names if n in ("nx", "ny", "nz", "nxx")]
    return {
        "n_rest": n_rest,
        "sh_degree": sh_degree_from_rest(n_rest),
        "missing_required": missing,
        "f_rest_contiguous": rest_ok,
        "normal_props": normals,
    }


def sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def safe_exp(v):
    try:
        return math.exp(v)
    except OverflowError:
        return float("inf")


def column_stats(col):
    finite = [v for v in col if math.isfinite(v)]
    bad = len(col) - len(finite)
    if not finite:
        return {"min": None, "max": None, "mean": None, "median": None, "nonfinite": bad}
    s = sorted(finite)
    return {"min": s[0], "max": s[-1], "mean": math.fsum(s) / len(s),
            "median": s[len(s) // 2], "p01": s[int(0.01 * (len(s) - 1))],
            "p99": s[int(0.99 * (len(s) - 1))], "nonfinite": bad}


def load_vertex_columns(f, hdr, wanted):
    """Read the vertex element; return {name: array('d' or 'f')} for wanted names."""
    fmt = hdr["format"]
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise PlyError(f"payload stats only supported for binary formats, got {fmt}")
    el = hdr["elements"][0]
    if el["name"] != "vertex":
        raise PlyError("first element is not 'vertex'")
    if el["has_list"]:
        raise PlyError("list properties in vertex element are not supported")
    props = el["props"]
    n = el["count"]
    stride = sum(PLY_TYPES[t][1] for t, _ in props)
    data = f.read(stride * n)
    if len(data) != stride * n:
        raise PlyError(f"truncated payload: expected {stride * n} bytes, got {len(data)}")
    names = [nm for _, nm in props]
    all_float = all(PLY_TYPES[t][0] == "f" for t, _ in props)
    cols = {}
    if all_float:
        a = array.array("f")
        a.frombytes(data)
        if (fmt == "binary_big_endian") != (sys.byteorder == "big"):
            a.byteswap()
        k = len(props)
        for w in wanted:
            if w in names:
                cols[w] = a[names.index(w)::k]
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        st = struct.Struct(endian + "".join(PLY_TYPES[t][0] for t, _ in props))
        rows = list(st.iter_unpack(data))
        for w in wanted:
            if w in names:
                i = names.index(w)
                cols[w] = array.array("d", (r[i] for r in rows))
    return cols


def analyze(path, header_only=False):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = read_header(f)
        out = {
            "file": path,
            "file_size_bytes": size,
            "format": hdr["format"],
            "header_size_bytes": hdr["header_size"],
            "comments": hdr["comments"],
            "elements": [{"name": e["name"], "count": e["count"],
                          "properties": [f"{t} {n}" for t, n in e["props"]]}
                         for e in hdr["elements"]],
        }
        if not hdr["elements"]:
            raise PlyError("no elements")
        v = hdr["elements"][0]
        out["vertex_count"] = v["count"]
        out.update(classify(v["props"]))
        stride = sum(PLY_TYPES[t][1] for t, _ in v["props"] if t != "list")
        out["vertex_stride_bytes"] = stride
        expected = hdr["header_size"] + stride * v["count"]
        out["expected_size_bytes"] = expected
        out["size_matches"] = (expected == size)
        if header_only or out["missing_required"]:
            return out
        wanted = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
        cols = load_vertex_columns(f, hdr, wanted)

    st = {}
    for axis in ("x", "y", "z"):
        st[axis] = column_stats(cols[axis])
    out["bbox_min"] = [st[a]["min"] for a in "xyz"]
    out["bbox_max"] = [st[a]["max"] for a in "xyz"]
    out["bbox_p01"] = [st[a]["p01"] for a in "xyz"]
    out["bbox_p99"] = [st[a]["p99"] for a in "xyz"]

    o = column_stats(cols["opacity"])
    out["opacity_raw"] = o
    out["opacity_activated"] = {k: (sigmoid(o[k]) if o.get(k) is not None else None)
                                for k in ("min", "p01", "median", "p99", "max")}
    sc = {}
    for i in range(3):
        s = column_stats(cols[f"scale_{i}"])
        sc[f"scale_{i}"] = {k: (safe_exp(s[k]) if s.get(k) is not None else None)
                            for k in ("min", "p01", "median", "p99", "max")}
        sc[f"scale_{i}"]["nonfinite"] = s["nonfinite"]
    out["scale_activated"] = sc

    r = [cols[f"rot_{i}"] for i in range(4)]
    norms = array.array("d", (math.sqrt(a * a + b * b + c * c + d * d)
                              for a, b, c, d in zip(*r)))
    out["quat_norm_raw"] = column_stats(norms)
    out["quat_zero_count"] = sum(1 for q in norms if q < 1e-12)

    dc = {}
    for i in range(3):
        s = column_stats(cols[f"f_dc_{i}"])
        dc["rgb"[i]] = {k: (0.5 + SH_C0 * s[k] if s.get(k) is not None else None)
                        for k in ("min", "median", "max")}
    out["dc_color_0p5_plus_C0_times_fdc"] = dc
    return out


def fmt_num(v):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def print_human(r):
    print(f"== {r['file']}")
    print(f"  size            : {r['file_size_bytes']} bytes ({r['file_size_bytes'] / 1e6:.1f} MB)")
    print(f"  format          : {r['format']}   header {r['header_size_bytes']} bytes")
    for c in r["comments"]:
        print(f"  comment         : {c}")
    for e in r["elements"]:
        print(f"  element         : {e['name']} x {e['count']} ({len(e['properties'])} properties)")
    props = [p.split()[-1] for p in r["elements"][0]["properties"]]
    types = sorted(set(p.split()[0] for p in r["elements"][0]["properties"]))
    rest = [p for p in props if p.startswith("f_rest_")]
    compact = [p for p in props if not p.startswith("f_rest_")]
    if rest:
        i = compact.index("f_dc_2") + 1 if "f_dc_2" in compact else len(compact)
        compact.insert(i, f"f_rest_0..{len(rest) - 1}")
    print(f"  properties      : {' '.join(compact)}   (types: {', '.join(types)})")
    print(f"  stride          : {r['vertex_stride_bytes']} bytes/gaussian")
    print(f"  SH degree       : {r['sh_degree']}  (f_rest count {r['n_rest']}, contiguous={r['f_rest_contiguous']})")
    print(f"  normals         : {r['normal_props'] or 'none'}")
    print(f"  size check      : expected {r['expected_size_bytes']} -> {'OK' if r['size_matches'] else 'MISMATCH'}")
    if r["missing_required"]:
        print(f"  MISSING REQUIRED: {r['missing_required']}")
    if "bbox_min" not in r:
        return
    print(f"  bbox min        : {[fmt_num(v) for v in r['bbox_min']]}")
    print(f"  bbox max        : {[fmt_num(v) for v in r['bbox_max']]}")
    print(f"  bbox p1..p99    : {[fmt_num(v) for v in r['bbox_p01']]} .. {[fmt_num(v) for v in r['bbox_p99']]}")
    oa = r["opacity_activated"]
    print(f"  opacity sigmoid : min {fmt_num(oa['min'])} p1 {fmt_num(oa['p01'])} med {fmt_num(oa['median'])} "
          f"p99 {fmt_num(oa['p99'])} max {fmt_num(oa['max'])}")
    for k, s in r["scale_activated"].items():
        print(f"  {k} exp     : min {fmt_num(s['min'])} p1 {fmt_num(s['p01'])} med {fmt_num(s['median'])} "
              f"p99 {fmt_num(s['p99'])} max {fmt_num(s['max'])}")
    q = r["quat_norm_raw"]
    print(f"  |rot| raw       : min {fmt_num(q['min'])} med {fmt_num(q['median'])} max {fmt_num(q['max'])} "
          f"(zero quats: {r['quat_zero_count']})")
    for ch, s in r["dc_color_0p5_plus_C0_times_fdc"].items():
        print(f"  DC color {ch}      : min {fmt_num(s['min'])} med {fmt_num(s['median'])} max {fmt_num(s['max'])}")
    nf = sum(v.get("nonfinite", 0) for v in [r["opacity_raw"], r["quat_norm_raw"]])
    print(f"  non-finite vals : {nf} (opacity+quat)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--header-only", action="store_true", help="skip payload statistics")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    results, rc = [], 0
    for p in args.files:
        try:
            r = analyze(p, args.header_only)
        except (PlyError, OSError) as e:
            print(f"== {p}\n  ERROR: {e}", file=sys.stderr)
            rc = 1
            continue
        results.append(r)
        if not args.json:
            print_human(r)
    if args.json:
        print(json.dumps(results, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
