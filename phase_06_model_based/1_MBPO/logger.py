from __future__ import annotations
import os
import csv
from typing import Dict

try:
    import wandb
    WANDB_FOUND = True
except ModuleNotFoundError:
    WANDB_FOUND = False

def init_logger(cfg):
    if cfg.logging.enabled and WANDB_FOUND:
        wandb.init(project=cfg.logging.wandb_project, config=cfg)
        return wandb
    
    os.makedirs(cfg.logging.csv_path, exist_ok=True)
    return None

def log_metrics(logger, metrics: Dict[str, float], *, step:int):
    if logger is None:
        csv_file = os.path.join("outputs", "metrics.csv")
        write_header = not os.path.exists(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["steps"] + list(metrics.keys()))
            
            writer.writerows([step] + list(metrics.values()))
    else:
        logger.log({**metrics, "step": step})