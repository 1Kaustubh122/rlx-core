from hydra import compose, initialize
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import main

def test_smoke_run():
    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["train.epochs=1", "env.sim_device=cpu", "logging.enabled=false"])
    main(cfg) 