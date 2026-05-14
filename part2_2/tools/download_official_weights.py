import argparse
import os
import urllib.request


URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
}


def report_hook(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100.0, downloaded * 100.0 / total_size)
    print(f"\rDownloading... {percent:5.1f}%", end="")


def main():
    parser = argparse.ArgumentParser(description="Download official Real-ESRGAN weights for part2_2.")
    parser.add_argument("model_name", choices=sorted(URLS))
    parser.add_argument("--out-dir", default="/home/fc/Coding/CV/part2_2/models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    url = URLS[args.model_name]
    out_path = os.path.join(args.out_dir, f"{args.model_name}.pth")
    if os.path.isfile(out_path):
        print(f"Already exists: {out_path}")
        return
    print(f"URL: {url}")
    print(f"Save to: {out_path}")
    urllib.request.urlretrieve(url, out_path, reporthook=report_hook)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

