"""Experiment logging: wandb (+ optional tensorboard), fail-safe by design.

Training must never crash because of logging. A wandb import/init failure (not
installed, not logged in, no network) is caught, downgraded to a printed warning,
and the run continues writing tensorboard (if enabled) and stdout only. Scalars
are fanned out to every active backend at a single monotonic ``global_step``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class RunLogger:
    """Fan training scalars out to wandb and/or tensorboard behind config toggles.

    :param cfg: resolved run config (its ``logging`` block drives the backends).
    :param out_dir: run output directory (tensorboard + wandb files live under it).
    :param run_name: display name; matches the run-dir basename.
    :param resume_id: a stored wandb run id to resume, or ``None`` for a fresh run.
    """

    def __init__(self, cfg: dict, out_dir: str | Path, run_name: str,
                 resume_id: Optional[str] = None):
        out_dir = Path(out_dir)
        self.tb = None
        if cfg["logging"]["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(out_dir / "tensorboard"))

        self.wandb = None
        self.run_id: Optional[str] = None
        wcfg = cfg["logging"]["wandb"]
        if wcfg["enabled"]:
            self._init_wandb(wcfg, cfg, out_dir, run_name, resume_id)

    def _init_wandb(self, wcfg: dict, cfg: dict, out_dir: Path,
                    run_name: str, resume_id: Optional[str]) -> None:
        try:
            import wandb
        except Exception as exc:  # not installed / broken install
            print(f"WARNING: wandb import failed ({exc}); continuing without wandb.")
            return
        init_kwargs = dict(
            project=wcfg["project"], entity=wcfg["entity"],
            tags=list(wcfg["tags"]), mode=wcfg["mode"],
            name=run_name, config=cfg, dir=str(out_dir),
        )
        try:
            if resume_id:
                try:
                    run = wandb.init(id=resume_id, resume="must", **init_kwargs)
                except Exception as exc:
                    print(f"WARNING: wandb resume of id={resume_id} failed ({exc}); "
                          f"starting a fresh wandb run.")
                    run = wandb.init(**init_kwargs)
            else:
                run = wandb.init(**init_kwargs)
        except Exception as exc:  # not logged in / no network
            print(f"WARNING: wandb.init failed ({exc}); continuing without wandb.")
            return
        self.wandb = wandb
        self.run_id = run.id
        print(f"wandb: run '{run.name}' id={run.id} mode={wcfg['mode']}")

    def log(self, scalars: dict, step: int) -> None:
        """Log a flat ``{tag: float}`` dict at ``step`` to all active backends."""
        step = int(step)
        if self.tb is not None:
            for key, val in scalars.items():
                self.tb.add_scalar(key, val, step)
        if self.wandb is not None:
            self.wandb.log(scalars, step=step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
