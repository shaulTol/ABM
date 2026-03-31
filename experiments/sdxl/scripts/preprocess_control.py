"""preprocess_control.py — Extract scribble/HED edges from an image for ControlNet conditioning.

Usage:
    python scripts/preprocess_control.py --input photo.jpg --output scribble.png
    python scripts/preprocess_control.py --input photo.jpg --output scribble.png --method hed
    python scripts/preprocess_control.py --input photo.jpg --output scribble.png --method canny
"""
import argparse
from PIL import Image


def main():
    p = argparse.ArgumentParser(description="Extract control image for ControlNet")
    p.add_argument("--input", type=str, required=True, help="Input image path")
    p.add_argument("--output", type=str, required=True, help="Output control image path")
    p.add_argument("--method", type=str, default="scribble",
                   choices=["scribble", "hed", "canny"],
                   help="Edge extraction method")
    p.add_argument("--size", type=int, default=512, help="Resize to this resolution")
    args = p.parse_args()

    img = Image.open(args.input).convert("RGB")
    img = img.resize((args.size, args.size))

    if args.method in ("scribble", "hed"):
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        use_scribble = args.method == "scribble"
        result = hed(img, scribble=use_scribble)
    elif args.method == "canny":
        import cv2
        import numpy as np
        img_np = np.array(img)
        edges = cv2.Canny(img_np, 100, 200)
        result = Image.fromarray(edges).convert("RGB")

    result.save(args.output)
    print(f"Saved {args.method} control image: {args.output} ({result.size})")


if __name__ == "__main__":
    main()
