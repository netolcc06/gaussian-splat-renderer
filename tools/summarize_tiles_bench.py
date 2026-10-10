"""Phase 6: one markdown table of splat-tiles benchmark JSONs (docs/notes/bench/<label>_<scene>_<renderer>.json).

  python tools/summarize_tiles_bench.py phase6_rect16 phase6_tight16 ... [--renderer splat-tiles]

Columns: entries M (last frame / max over the orbit), list-build ms (scans .. ranges, i.e. everything
between the preprocess and the render), render ms, preprocess ms, duplicate, tile sort, GPU frame,
FPS (CPU frame), list MB, registered total MB.
"""
import json
import sys
from pathlib import Path

SCENES = ['utsuho_plush', 'train_7k', 'bonsai_30k']
LIST_STAGES = ('scan: visible', 'compact', 'depth sort (total)', 'scan: tiles', 'tile setup', 'duplicate',
               'tile sort (total)', 'tile ranges')


def stage(g, prefix):
    for k, v in g.items():
        if k.startswith(prefix):
            return v['avg']
    return 0.0


def main(argv):
    renderer = 'splat-tiles'
    labels = []
    it = iter(argv)
    for a in it:
        if a == '--renderer':
            renderer = next(it)
        else:
            labels.append(a)
    root = Path(__file__).resolve().parent.parent / 'docs' / 'notes' / 'bench'
    print('| config | scene | entries M (last / max) | list build ms | duplicate | tile sort | render | preprocess '
          '| GPU frame ms | FPS | list MB | total MB | coverage mismatches |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for label in labels:
        for s in SCENES:
            p = root / f'{label}_{s}_{renderer}.json'
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            g, r = j['gpu_ms'], j.get('renderer_settings', {})
            lb = sum(stage(g, k) for k in LIST_STAGES)
            print('| {} | {} | {:.2f} / {:.2f} M | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
                label, s, r.get('duplicates_last_frame', 0) / 1e6, r.get('max_duplicates', 0) / 1e6, lb,
                stage(g, 'duplicate'), stage(g, 'tile sort (total)'), stage(g, 'render tiles'), stage(g, 'preprocess'),
                g['frame (total)']['avg'], j['fps_avg'], r.get('list_mb', 0), j['gpu_memory']['total_registered_mb'],
                r.get('coverage_mismatches', '-')))


if __name__ == '__main__':
    main(sys.argv[1:])
