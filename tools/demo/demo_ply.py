#!/usr/bin/env python3
"""Small, dependency-free helpers for the web demo scene (tools/demo/make_demo_scene.ps1).

Subcommands (all deterministic, pure standard library, run with `python -I`):

  canon <in.ply> <out.ply> [comment ...]
      Rewrite a binary little-endian float32 3DGS PLY with the properties in the INRIA
      `construct_list_of_attributes` order, without normals:
          x y z f_dc_0..2 f_rest_0..(n-1) opacity scale_0..2 rot_0..3
      splat-transform writes `x y z rot_* scale_* opacity f_dc_* f_rest_*`; our loader matches
      by name so either works, but the canonical order is what other 3DGS tools expect. Values
      are copied bit-exactly (still pre-activation: logit opacity, log scale, w-first quat).
      Extra arguments become `comment` header lines (provenance/attribution).

  prune <in.ply> <out.ply> <N>
      Keep the N most "important" gaussians, importance = sigmoid(opacity) * exp(s0+s1+s2)
      (alpha x ellipsoid volume / (4/3 pi)). Ties are broken by file index; survivors keep
      their original file order. Used only as a comparison baseline for splat-transform's
      --decimate (see docs/web/DEMO_SCENE.md). Output property order = input order.

  gzip <in> <out.gz>
      Deterministic gzip (level 9, mtime 0, no file name in the header) so that rebuilding
      produces byte-identical output. The web page inflates it with DecompressionStream.

The input is treated as untrusted data: only parsed, never executed.
"""
import array
import gzip
import math
import os
import sys


def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line or len(header) > 65536:
                raise SystemExit(f"{path}: no end_header")
            header += line
        payload = f.read()
    lines = header.decode("ascii").split("\n")
    if lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise SystemExit(f"{path}: not a binary little endian PLY")
    count, props, in_vertex = None, [], False
    for ln in lines:
        tok = ln.split()
        if tok[:2] == ["element", "vertex"]:
            count, in_vertex = int(tok[2]), True
        elif tok and tok[0] == "element":
            in_vertex = False
        elif tok and tok[0] == "property" and in_vertex:
            if tok[1] != "float":
                raise SystemExit(f"{path}: property {tok[-1]} is {tok[1]}, expected float")
            props.append(tok[2])
    stride = len(props)
    if count is None or len(payload) != count * stride * 4:
        raise SystemExit(f"{path}: size mismatch ({len(payload)} bytes, {count} x {stride} floats)")
    data = array.array("f")
    data.frombytes(payload)
    if sys.byteorder != "little":
        data.byteswap()
    return props, count, data


def write_ply(path, props, count, data, comment):
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in comment]
    header += [f"element vertex {count}"] + [f"property float {p}" for p in props] + ["end_header", ""]
    out = data
    if sys.byteorder != "little":
        out = array.array("f", data)
        out.byteswap()
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(out.tobytes())


def canonical_order(props):
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p[7:]))
    order = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + rest + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [p for p in order if p not in props]
    if missing:
        raise SystemExit(f"missing properties: {missing}")
    dropped = [p for p in props if p not in order]
    return order, dropped


def cmd_canon(src, dst, comments):
    props, n, data = read_ply(src)
    order, dropped = canonical_order(props)
    s_in, s_out = len(props), len(order)
    out = array.array("f", bytes(4 * n * s_out))
    for k, name in enumerate(order):
        out[k::s_out] = data[props.index(name)::s_in]
    write_ply(dst, order, n, out, list(comments) + ["property order canonicalized by tools/demo/demo_ply.py"])
    print(f"canon: {n} gaussians, {s_in} -> {s_out} properties, dropped {dropped or 'none'}")


def cmd_prune(src, dst, keep):
    props, n, data = read_ply(src)
    s = len(props)
    io, i0, i1, i2 = (props.index(p) for p in ("opacity", "scale_0", "scale_1", "scale_2"))
    op, s0, s1, s2 = data[io::s], data[i0::s], data[i1::s], data[i2::s]

    def importance(i):
        o = op[i]
        a = 1.0 / (1.0 + math.exp(-o)) if o > -80 else 0.0
        return a * math.exp(min(s0[i] + s1[i] + s2[i], 80.0))

    keep = min(keep, n)
    order = sorted(range(n), key=lambda i: (-importance(i), i))[:keep]
    order.sort()
    out = array.array("f")
    for i in order:
        out.extend(data[i * s:(i + 1) * s])
    write_ply(dst, props, keep, out, [f"pruned to {keep} of {n} by opacity x volume (tools/demo/demo_ply.py)"])
    print(f"prune: kept {keep} of {n}")


def cmd_gzip(src, dst):
    with open(src, "rb") as f:
        raw = f.read()
    with open(dst, "wb") as out:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=out, mtime=0) as gz:
            gz.write(raw)
    print(f"gzip: {len(raw)} -> {os.path.getsize(dst)} bytes")


def main(argv):
    if len(argv) >= 3 and argv[0] == "canon":
        cmd_canon(argv[1], argv[2], argv[3:])
    elif len(argv) == 4 and argv[0] == "prune":
        cmd_prune(argv[1], argv[2], int(argv[3]))
    elif len(argv) == 3 and argv[0] == "gzip":
        cmd_gzip(argv[1], argv[2])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
