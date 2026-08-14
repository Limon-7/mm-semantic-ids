"""Experiment configuration.

An experiment file under configs/exp/ names a dataset, a split, an ID recipe and a model. It is
merged on top of configs/defaults.yaml, and `key=value` arguments on the command line are merged
last, so any field can be overridden for a single run.
"""

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(exp: str | None, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_DIR / "defaults.yaml")
    if exp:
        path = Path(exp)
        if not path.exists():
            path = CONFIG_DIR / "exp" / f"{exp}.yaml"
        exp_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, exp_cfg)
        if "name" not in exp_cfg:
            cfg.name = path.stem  # the experiment is called after its file
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def config_hash(cfg: DictConfig) -> str:
    return hashlib.sha1(OmegaConf.to_yaml(cfg, sort_keys=True).encode()).hexdigest()[:10]
