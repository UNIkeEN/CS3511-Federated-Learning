import argparse
from omegaconf import OmegaConf
from pipeline import OfflinePipeline, OnlinePipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the config file")
    args, extras = parser.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if cfg.mode == "offline":
        pipe = OfflinePipeline(cfg)

    if cfg.mode == "online":
        pipe = OnlinePipeline(cfg, args.config)
    
    pipe.train()