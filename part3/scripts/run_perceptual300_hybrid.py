import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate perceptual300 conservative hybrid outputs.")
    parser.add_argument("--config", default="part3/configs/perceptual300_conservative.yaml")
    args = parser.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config)
    for seq in cfg["sequences"]:
        seq_cfg = {
            "paths": {
                "basic_dir": cfg["inputs"]["basic_template"].format(seq=seq),
                "basic_panel_index": 0,
                "basic_panel_count": 1,
                "lr_dir": str(Path(cfg["inputs"]["lr_root"]) / seq),
                "lr_panel_index": 0,
                "generative_dir": cfg["inputs"]["generative_template"].format(seq=seq),
                "generative_suffix": "",
                "output_dir": str(Path(cfg["output"]["result_root"]) / seq),
            },
            "adaptive": cfg["adaptive"],
            "output": {
                "save_grids": cfg["output"].get("save_grids", False),
                "save_maps": cfg["output"].get("save_maps", True),
                "export_video": cfg["output"].get("export_video", True),
                "fps": cfg["output"].get("fps", 30),
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=f"_perceptual300_{seq}.yaml", delete=False) as f:
            yaml.safe_dump(seq_cfg, f, sort_keys=False)
            temp_path = f.name
        run([sys.executable, "part3/scripts/run_adaptive_hybrid.py", "--config", temp_path])


if __name__ == "__main__":
    main()
