"""Checks that the WGSL ports in shaders/wgsl/ are in sync with their GLSL sources (web phase W2).

shaders/wgsl/PORTS.md has a table: GLSL source(s) -> WGSL port, the SHA-256 (first 16 hex digits,
computed with LF line endings so the checkout's line-ending style does not matter) of each GLSL file
at the time its port was last reviewed, and the deliberate differences. This script recomputes the
hashes and fails if a GLSL file changed since (DESIGN.md section 4.3): update the port, or only the
hash if the change does not matter on the web (`--update` rewrites every hash column).

It also checks that every listed WGSL file exists and names its GLSL source in its first lines
("Port of shaders/<file> ..."), and that every .wgsl file in shaders/wgsl/ appears in the table.

  python tools/web/check_shader_sync.py            exit 0 = in sync, 1 = drift
  python tools/web/check_shader_sync.py --update   rewrite the hashes (after reviewing the ports)
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHADERS = ROOT / "shaders"
MANIFEST = SHADERS / "wgsl" / "PORTS.md"
ROW = re.compile(r"^\|\s*`(?P<glsl>[^`]+)`\s*\|\s*`(?P<wgsl>[^`]+)`\s*\|\s*`(?P<hash>[0-9a-f]{16}|-)`\s*\|(?P<rest>.*)$")


def glsl_hash(name: str) -> str:
    data = (SHADERS / name).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    update = "--update" in sys.argv[1:]
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    problems, rows, listed = [], 0, set()
    out = []
    for line in lines:
        m = ROW.match(line)
        if not m:
            out.append(line)
            continue
        rows += 1
        glsl, wgsl, recorded = m["glsl"], m["wgsl"], m["hash"]
        listed.add(wgsl)
        wgsl_path = SHADERS / "wgsl" / wgsl
        if not wgsl_path.exists():
            problems.append(f"{wgsl}: listed but missing")
        if glsl == "(new)":
            out.append(line)
            continue
        if not (SHADERS / glsl).exists():
            problems.append(f"{glsl}: listed GLSL source does not exist")
            out.append(line)
            continue
        actual = glsl_hash(glsl)
        if wgsl_path.exists():
            head = "\n".join(wgsl_path.read_text(encoding="utf-8").splitlines()[:6])
            stem = glsl.split(".")[0]
            if glsl not in head and stem not in head:
                problems.append(f"{wgsl}: its header does not name its source {glsl}")
        if actual != recorded:
            if update:
                line = line.replace(f"`{recorded}`", f"`{actual}`", 1)
            else:
                problems.append(f"{glsl} changed since its WGSL port {wgsl} was reviewed "
                                f"(recorded {recorded}, now {actual}): update the port, then run with --update")
        out.append(line)
    for f in sorted((SHADERS / "wgsl").glob("*.wgsl")):
        if f.name not in listed:
            problems.append(f"{f.name}: not in PORTS.md")
    if update:
        MANIFEST.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"updated {rows} rows in {MANIFEST.relative_to(ROOT)}")
        return 0
    for p in problems:
        print("DRIFT:", p)
    if problems:
        print(f"{len(problems)} problem(s) in {rows} rows")
        return 1
    print(f"shader ports in sync: {rows} rows, {len(listed)} WGSL files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
