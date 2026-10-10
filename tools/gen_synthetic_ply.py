#!/usr/bin/env python3
"""Generate synthetic 3D Gaussian Splatting .ply files in the standard INRIA format.

Output layout is byte-compatible with graphdeco-inria/gaussian-splatting `save_ply`:
binary_little_endian, one `vertex` element, all `float` (float32) properties in order

    x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 f_rest_0..f_rest_{N-1}
    opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3

with N = 3 * ((deg+1)^2 - 1) (0, 9, 24, 45 for degree 0..3). Values are stored
PRE-activation, exactly like trained files:
    opacity = logit(alpha)          (renderer applies sigmoid)
    scale_i = log(sigma_i)          (renderer applies exp)
    rot     = (w, x, y, z), possibly NOT unit length (renderer normalizes)
    f_dc    = (rgb - 0.5) / C0      (renderer: rgb = 0.5 + C0*f_dc + higher bands, clamp >= 0)
    f_rest  channel-major: f_rest[c*(K-1) + (k-1)] = coefficient k (1..K-1) of channel c (R,G,B)

Patterns (all deterministic):
    single   one isotropic gaussian at the origin (sigma 0.5, alpha 0.8, orange)
    grid     a k*k*k grid of anisotropic gaussians, colour from position, rotations
             varying per cell (stretched 8:2:1 along their local axes)
    random   uniform cloud in a sphere of radius 2, fixed --seed; random log-scales,
             unnormalized random quaternions, random SH coefficients
    overlap  four large semi-transparent (alpha 0.5) gaussians R,G,B,white stacked
             along z with lateral offsets -> tests back-to-front blending order
    shdir    6 gaussians whose colour depends on view direction through band-1/2/3
             SH only (DC = grey) -> tests SH evaluation and the direction convention

Scene convention: everything lives within ~[-2.5, 2.5]^3 around the origin; a camera
at distance ~6-8 looking at the origin frames it.

Usage:
    python -I tools/gen_synthetic_ply.py --pattern grid --sh-degree 3 --count 1000 -o out.ply
    python -I tools/gen_synthetic_ply.py --all          # writes the standard set into data/synthetic/
    python -I tools/gen_synthetic_ply.py --expected-json out.json   # expected loader values (tests)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys

SH_C0 = 0.28209479177387814
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic")


def n_coeffs(deg):
    return (deg + 1) ** 2


def logit(a):
    a = min(max(a, 1e-6), 1 - 1e-6)
    return math.log(a / (1 - a))


def rgb_to_dc(rgb):
    return [(c - 0.5) / SH_C0 for c in rgb]


def quat_axis_angle(axis, angle, scale=1.0):
    """(w, x, y, z) quaternion; `scale` deliberately de-normalizes it."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    s = math.sin(angle / 2) / n
    return [scale * math.cos(angle / 2), scale * ax * s, scale * ay * s, scale * az * s]


class Gaussian:
    __slots__ = ("pos", "rgb", "alpha", "sigma", "quat", "rest")

    def __init__(self, pos, rgb, alpha, sigma, quat, rest=None):
        self.pos = pos            # [x, y, z]
        self.rgb = rgb            # DC colour in [0,1] (what you see for deg 0)
        self.alpha = alpha        # activated opacity in (0,1)
        self.sigma = sigma        # activated per-axis std-dev (world units)
        self.quat = quat          # (w, x, y, z), any non-zero length
        self.rest = rest          # rest[c][k-1] for k in 1..15, or None (zeros)


def property_names(deg, with_normals=True):
    names = ["x", "y", "z"]
    if with_normals:
        names += ["nx", "ny", "nz"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * (n_coeffs(deg) - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def record_row(g, deg, with_normals=True):
    """The float values of one PLY record, in property_names() order (float64)."""
    rest_per_ch = n_coeffs(deg) - 1
    row = list(g.pos)
    if with_normals:
        row += [0.0, 0.0, 0.0]   # INRIA always writes zeros here
    row += rgb_to_dc(g.rgb)
    for c in range(3):           # channel-major f_rest
        src = g.rest[c] if g.rest else []
        for k in range(rest_per_ch):
            row.append(src[k] if k < len(src) else 0.0)
    row.append(logit(g.alpha))
    row += [math.log(s) for s in g.sigma]
    row += list(g.quat)
    return row


def write_ply(path, gaussians, deg, with_normals=True, comment=None):
    names = property_names(deg, with_normals)
    header = ["ply", "format binary_little_endian 1.0"]
    if comment:
        header.append(f"comment {comment}")
    header.append(f"element vertex {len(gaussians)}")
    header += [f"property float {n}" for n in names]
    header.append("end_header")
    rec = struct.Struct("<" + "f" * len(names))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for g in gaussians:
            f.write(rec.pack(*record_row(g, deg, with_normals)))
    return path


def to_f32(v):
    """Round a Python float to the float32 value the PLY file stores."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def expected_values(pattern, deg, count, seed, normals):
    """What a loader must read back from the file generate() writes (for tests).

    - sample rows: full records (float32-exact) at a few indices
    - sums: float64 sums over ALL records of each non-SH property and of all f_rest
      values, so a test can check every record without storing them all.
    """
    rng = random.Random(seed)
    gs = PATTERNS[pattern](count, rng)
    names = property_names(deg, normals)
    n = len(gs)
    sample_idx = sorted({0, 1 % n, n // 2, n - 1})
    samples = []
    sums = {k: 0.0 for k in names if not k.startswith("f_rest_") and k not in ("nx", "ny", "nz")}
    sums["f_rest_all"] = 0.0
    for i, g in enumerate(gs):
        row = [to_f32(v) for v in record_row(g, deg, normals)]
        for name, v in zip(names, row):
            if name.startswith("f_rest_"):
                sums["f_rest_all"] += v
            elif name in sums:
                sums[name] += v
        if i in sample_idx:
            samples.append({"index": i, "values": dict(zip(names, row))})
    return {
        "pattern": pattern, "count": n, "sh_degree": deg, "normals": normals, "seed": seed,
        "property_names": names, "samples": samples, "sums": sums,
    }


# ----------------------------------------------------------------------------- patterns

def pattern_single(count, rng):
    return [Gaussian([0.0, 0.0, 0.0], [1.0, 0.5, 0.1], 0.8, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0])]


def hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def pattern_grid(count, rng):
    k = max(1, round(count ** (1 / 3)))
    out = []
    span = 4.0
    step = span / k
    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    idx = 0
    for iz in range(k):
        for iy in range(k):
            for ix in range(k):
                u = [(ix + 0.5) / k, (iy + 0.5) / k, (iz + 0.5) / k]
                pos = [-span / 2 + (i + 0.5) * step for i in (ix, iy, iz)]
                rgb = [0.15 + 0.85 * c for c in u]
                base = 0.22 * step
                sigma = [base, base * 0.25, base * 0.125]   # 8 : 2 : 1 anisotropy
                angle = (idx * 0.61803398875 % 1.0) * math.pi   # golden-ratio spread of angles
                quat = quat_axis_angle(axes[idx % len(axes)], angle)
                alpha = 0.6 + 0.35 * ((idx * 7) % 5) / 4
                # small deterministic view-dependence for SH tests: band-1 lobe
                rest = [[0.0] * 15 for _ in range(3)]
                rest[idx % 3][2] = 0.4      # coefficient k=3 (x-lobe) of one channel
                out.append(Gaussian(pos, rgb, alpha, sigma, quat, rest))
                idx += 1
    return out


def random_unit(rng):
    while True:
        v = [rng.uniform(-1, 1) for _ in range(3)]
        n2 = sum(c * c for c in v)
        if 1e-6 < n2 <= 1:
            return v, math.sqrt(n2)


def pattern_random(count, rng):
    out = []
    for _ in range(count):
        v, _n = random_unit(rng)
        r = 2.0 * rng.random() ** (1 / 3)
        dirn = [c / _n for c in v]
        pos = [r * c for c in dirn]
        rgb = [rng.random() for _ in range(3)]
        alpha = rng.uniform(0.05, 0.99)
        sigma = [math.exp(rng.uniform(math.log(0.005), math.log(0.12))) for _ in range(3)]
        quat = [rng.gauss(0, 1) for _ in range(4)]       # unnormalized on purpose
        rest = [[rng.gauss(0, 0.15) for _ in range(15)] for _ in range(3)]
        out.append(Gaussian(pos, rgb, alpha, sigma, quat, rest))
    return out


def pattern_overlap(count, rng):
    # Back (z = -1.5) to front (z = +1.5) in world space; camera on +z looking at -z sees
    # white in front. Lateral offsets make each pair overlap only partially.
    specs = [
        ([-0.4, -0.3, -1.5], [1.0, 0.0, 0.0]),
        ([0.4, -0.3, -0.5], [0.0, 1.0, 0.0]),
        ([0.0, 0.4, 0.5], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.5], [1.0, 1.0, 1.0]),
    ]
    out = []
    for i, (p, c) in enumerate(specs):
        quat = quat_axis_angle((0, 0, 1), i * math.pi / 8, scale=2.0)  # also non-unit
        out.append(Gaussian(p, c, 0.5, [0.7, 0.45, 0.3], quat))
    return out


def pattern_shdir(count, rng):
    """Grey DC + one SH basis function per gaussian, on a ring in the z=0 plane.

    Coefficient index k (1..15) in standard 3DGS order:
      1: -C1*y   2: C1*z   3: -C1*x   4..8: band 2   9..15: band 3.
    With degree < required band, the gaussian is plain grey (useful sanity check).
    """
    ks = [1, 2, 3, 4, 6, 9]
    out = []
    for i, k in enumerate(ks):
        a = 2 * math.pi * i / len(ks)
        pos = [1.8 * math.cos(a), 1.8 * math.sin(a), 0.0]
        rest = [[0.0] * 15 for _ in range(3)]
        rest[0][k - 1] = 1.2      # red channel  +lobe
        rest[2][k - 1] = -1.2     # blue channel -lobe
        out.append(Gaussian(pos, [0.5, 0.5, 0.5], 0.9, [0.35, 0.35, 0.35], [1, 0, 0, 0], rest))
    return out


PATTERNS = {
    "single": pattern_single,
    "grid": pattern_grid,
    "random": pattern_random,
    "overlap": pattern_overlap,
    "shdir": pattern_shdir,
}

# The standard set written by --all: (pattern, sh_degree, count, normals)
STANDARD_SET = [
    ("single", 0, 1, True),
    ("single", 3, 1, True),
    ("grid", 0, 512, True),
    ("grid", 3, 512, True),
    ("random", 1, 10000, True),
    ("random", 3, 100000, True),
    ("random", 2, 2000, False),     # variant without nx/ny/nz
    ("overlap", 0, 4, True),
    ("shdir", 3, 6, True),
]


def generate(pattern, deg, count, seed, out, normals=True):
    rng = random.Random(seed)
    gs = PATTERNS[pattern](count, rng)
    comment = f"synthetic pattern={pattern} sh_degree={deg} seed={seed} generator=tools/gen_synthetic_ply.py"
    write_ply(out, gs, deg, normals, comment)
    return out, len(gs)


def default_name(pattern, deg, count, normals):
    n = "" if pattern in ("single", "overlap", "shdir") else f"_{count}"
    return f"{pattern}{n}_sh{deg}{'' if normals else '_nonormals'}.ply"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="grid")
    ap.add_argument("--sh-degree", type=int, choices=[0, 1, 2, 3], default=3)
    ap.add_argument("--count", type=int, default=1000,
                    help="number of gaussians (grid rounds to k^3; ignored by single/overlap/shdir)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-normals", action="store_true", help="omit nx/ny/nz (valid variant)")
    ap.add_argument("-o", "--output", help="output path (default: data/synthetic/<auto>.ply)")
    ap.add_argument("--all", action="store_true", help="write the standard test set to data/synthetic/")
    ap.add_argument("--expected-json", metavar="PATH",
                    help="write the values a loader must read from the standard set (keyed by file name) "
                         "to PATH and exit without writing any .ply (used by the C++ loader tests)")
    args = ap.parse_args(argv)

    if args.expected_json:
        out = {}
        for p, d, c, nrm in STANDARD_SET:
            out[default_name(p, d, c, nrm)] = expected_values(p, d, c, args.seed, nrm)
        os.makedirs(os.path.dirname(os.path.abspath(args.expected_json)), exist_ok=True)
        with open(args.expected_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.expected_json} ({len(out)} files)")
        return 0

    jobs = []
    if args.all:
        for p, d, c, nrm in STANDARD_SET:
            jobs.append((p, d, c, nrm, os.path.join(DEFAULT_OUT_DIR, default_name(p, d, c, nrm))))
    else:
        nrm = not args.no_normals
        out = args.output or os.path.join(DEFAULT_OUT_DIR, default_name(args.pattern, args.sh_degree, args.count, nrm))
        jobs.append((args.pattern, args.sh_degree, args.count, nrm, out))

    for p, d, c, nrm, out in jobs:
        path, n = generate(p, d, c, args.seed, out, nrm)
        print(f"wrote {path}  ({n} gaussians, SH degree {d}, {os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
