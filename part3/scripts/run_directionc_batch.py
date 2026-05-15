import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_CONFIG = REPO_ROOT / "part3" / "configs" / "directionc_val000_006_batch.yaml"


def run(cmd):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_batch_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seq_basic_path(batch_cfg: dict, seq: str):
    inputs = batch_cfg["inputs"]
    if seq in set(inputs.get("compare_sequences", [])):
        return inputs["basic_compare_template"].format(seq=seq), 1, 3
    return inputs["basic_frames_template"].format(seq=seq), 0, 1


def make_sequence_config(batch_cfg: dict, variant: str, seq: str) -> dict:
    variant_cfg = batch_cfg["variants"][variant]
    basic_dir, basic_panel_index, basic_panel_count = seq_basic_path(batch_cfg, seq)
    inputs = batch_cfg["inputs"]
    return {
        "paths": {
            "basic_dir": basic_dir,
            "basic_panel_index": basic_panel_index,
            "basic_panel_count": basic_panel_count,
            "lr_dir": str(Path(inputs["lr_root"]) / seq),
            "lr_panel_index": 0,
            "generative_dir": inputs["generative_template"].format(seq=seq),
            "generative_suffix": "",
            "output_dir": str(Path(variant_cfg["result_root"]) / seq),
        },
        "adaptive": variant_cfg["adaptive"],
        "output": batch_cfg["output"],
    }


def write_temp_config(batch_cfg: dict, variant: str, seq: str) -> str:
    cfg = make_sequence_config(batch_cfg, variant, seq)
    handle = tempfile.NamedTemporaryFile("w", suffix=f"_{variant}_{seq}.yaml", delete=False)
    with handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return handle.name


def result_dir(batch_cfg: dict, variant: str, seq: str) -> Path:
    return Path(batch_cfg["variants"][variant]["result_root"]) / seq


def metrics_dir(batch_cfg: dict, variant: str, seq: str) -> Path:
    return Path(batch_cfg["variants"][variant]["metrics_root"]) / seq


def aggregate_metrics(batch_cfg: dict, variant: str, seqs, all_seqs):
    root = Path(batch_cfg["variants"][variant]["metrics_root"])
    rows = []
    for seq in seqs:
        summary_path = root / seq / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing metrics: {summary_path}")
        with open(summary_path, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        rows.append(
            {
                "sequence": seq,
                "frames": int(row["frames"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
            }
        )

    total_frames = sum(row["frames"] for row in rows)
    overall = {
        "sequence": f"overall_{seqs[0]}_{seqs[-1]}",
        "frames": total_frames,
        "psnr": sum(row["psnr"] * row["frames"] for row in rows) / total_frames,
        "ssim": sum(row["ssim"] * row["frames"] for row in rows) / total_frames,
    }

    os.makedirs(root, exist_ok=True)
    summary_name = "summary.csv" if list(seqs) == list(all_seqs) else f"summary_{seqs[0]}_{seqs[-1]}.csv"
    summary_md_name = "summary.md" if list(seqs) == list(all_seqs) else f"summary_{seqs[0]}_{seqs[-1]}.md"

    with open(root / summary_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "frames", "psnr", "ssim"])
        writer.writeheader()
        writer.writerow(overall)
        writer.writerows(rows)

    with open(root / summary_md_name, "w", encoding="utf-8") as f:
        f.write("| Sequence | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in [overall] + rows:
            f.write(
                f"| {row['sequence']} | {row['frames']} | "
                f"{row['psnr']:.4f} | {row['ssim']:.4f} |\n"
            )

    print(f"Saved aggregate metrics: {root / summary_name}")
    return overall, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for final Direction C variants.")
    parser.add_argument("--config", default=str(DEFAULT_BATCH_CONFIG), help="Batch config yaml.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seqs", nargs="+", default=None)
    parser.add_argument("--run-hybrid", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--showcase", action="store_true")
    parser.add_argument("--frame", default="00000049")
    return parser.parse_args()


def main():
    args = parse_args()
    batch_cfg = load_batch_config(Path(args.config))
    if args.variant not in batch_cfg["variants"]:
        choices = ", ".join(sorted(batch_cfg["variants"]))
        raise SystemExit(f"Unknown variant '{args.variant}'. Available variants: {choices}")
    if not (args.run_hybrid or args.evaluate or args.showcase):
        raise SystemExit("Nothing to do. Pass --run-hybrid, --evaluate, or --showcase.")

    all_seqs = batch_cfg["sequences"]
    seqs = args.seqs or all_seqs
    for seq in seqs:
        print(f"=== {args.variant} seq {seq} ===", flush=True)
        if args.run_hybrid:
            temp_config = write_temp_config(batch_cfg, args.variant, seq)
            run(
                [
                    sys.executable,
                    "part3/scripts/run_adaptive_hybrid.py",
                    "--config",
                    temp_config,
                ]
            )

        if args.evaluate:
            run(
                [
                    sys.executable,
                    "part2_2/src/evaluate.py",
                    "--pred",
                    str(result_dir(batch_cfg, args.variant, seq) / "frames"),
                    "--gt",
                    str(Path(batch_cfg["inputs"]["gt_root"]) / seq),
                    "--output",
                    str(metrics_dir(batch_cfg, args.variant, seq)),
                    "--pred-suffix",
                    "",
                ]
            )

        if args.showcase:
            run(
                [
                    sys.executable,
                    "part3/scripts/make_directionc_showcase.py",
                    "--result-dir",
                    str(result_dir(batch_cfg, args.variant, seq)),
                    "--frame",
                    args.frame,
                ]
            )

    if args.evaluate:
        aggregate_metrics(batch_cfg, args.variant, seqs, all_seqs)


if __name__ == "__main__":
    main()
