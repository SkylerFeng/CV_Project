import argparse
import os
import subprocess


def ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Resize a video to a target resolution with ffmpeg.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--width", type=int, required=True, help="Target width")
    parser.add_argument("--height", type=int, required=True, help="Target height")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF. Lower is higher quality.")
    parser.add_argument("--preset", default="slow", help="x264 preset")
    parser.add_argument("--no-audio", action="store_true", help="Drop audio")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    ensure_parent(args.output)

    vf = (
        f"scale={args.width}:{args.height}:"
        "flags=lanczos,setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        args.input,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if args.no_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "copy"])
    cmd.append(args.output)

    run_cmd(cmd)
    print(f"Saved resized video to: {args.output}")


if __name__ == "__main__":
    main()

