"""Browse every run's test-set predictions next to the GT and the frozen body in 3D.

One viser server for all runs under an output directory that carry a
``predictions/`` dump (``scripts/predict_test.py``): pick the run and the scene
in the sidebar, scrub or play the frames, and switch between the two viewing
regimes — ``camera`` (the bodies exactly as the model outputs them in the
frame's camera, the GT lifted into it) and ``world`` (everything lifted into
the metric world with the corpus extrinsics, camera path and gravity shown).
Each of the three bodies (predicted / GT / frozen SAM 3D) has its own mesh and
skeleton toggles; a slider sets the mesh opacity; the sidebar video pane plays
the source frames in sync. Each run's corpus (ClimbingVideos or BEDLAM2_our) is
the dataset of its ``config.yaml``; a BEDLAM scene has no scene cloud and no
frozen SAM 3D body.

``--wild <root>`` browses the other kind of dump instead: in-the-wild videos
processed by ``scripts/predict_reconstruction.py`` into ``<root>/<stem>/``. Those
carry only the predicted body (no GT, no frozen refit, no scene cloud), the world
is oriented with the model's own predicted gravity, and the video pane decodes the
dump's ``source_video``.

    python scripts/view_results.py --output output_6                 # every run, port 8090
    python scripts/view_results.py --output output_7 --run bedlam_frames35_20260920_124113
    python scripts/view_results.py --wild                             # the wild trees
    python scripts/view_results.py --no-video --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viewer import view_results                          # noqa: E402

WILD_ROOT = Path("/home/rikhat.akizhanov/better/data/willd_videos/out")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("output_7"),
                        help="directory holding the <run>/predictions dumps")
    parser.add_argument("--run", default=None, help="run directory name to open first")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="cuda", help="device of the SMPL-X FK")
    parser.add_argument("--no-video", action="store_true", help="skip the sidebar video pane")
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--wild", type=Path, nargs="?", const=WILD_ROOT, default=None,
                        help="browse the in-the-wild prediction trees under this root")
    args = parser.parse_args()
    view_results(args.output, port=args.port, device=args.device,
                 video=not args.no_video, run=args.run, opacity=args.opacity,
                 wild=args.wild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
