"""Browse every run's test-set predictions next to the GT and the frozen body in 3D.

One viser server for all runs under ``output/`` that carry a
``predictions/`` dump (``scripts/predict_test.py``): pick the run and the scene
in the sidebar, scrub or play the frames, and switch between the two viewing
regimes — ``camera`` (the bodies exactly as the model outputs them in the
frame's camera, the GT lifted into it) and ``world`` (everything lifted into
the metric world with the corpus extrinsics, camera path and gravity shown).
Each of the three bodies (predicted / GT / frozen SAM 3D) has its own mesh and
skeleton toggles; a slider sets the mesh opacity; the sidebar video pane plays
the source frames in sync.

    python scripts/view_results.py                       # every run, port 8090
    python scripts/view_results.py --run tb_projzero_20260904_225421 --port 8091
    python scripts/view_results.py --no-video --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viewer import view_results                          # noqa: E402

CORPUS = Path("/data3/rikhat.akizhanov/better/data/ClimbingVideos")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="directory holding the <run>/predictions dumps")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--run", default=None, help="run directory name to open first")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="cuda", help="device of the SMPL-X FK")
    parser.add_argument("--no-video", action="store_true", help="skip the sidebar video pane")
    parser.add_argument("--opacity", type=float, default=0.85)
    args = parser.parse_args()
    view_results(args.output, args.corpus, port=args.port, device=args.device,
                 video=not args.no_video, run=args.run, opacity=args.opacity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
