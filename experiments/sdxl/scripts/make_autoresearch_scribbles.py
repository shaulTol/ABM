"""Generate scribble images for the 3 autoresearch experiments."""

from PIL import Image, ImageDraw, ImageFilter
import os


def make_scribble_face_outline(size=512, lw=3):
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)
    d.ellipse([140, 80, 372, 400], outline="white", width=lw)
    d.ellipse([185, 185, 225, 210], outline="white", width=lw)
    d.ellipse([287, 185, 327, 210], outline="white", width=lw)
    d.line([256, 225, 245, 285], fill="white", width=lw)
    d.line([245, 285, 265, 285], fill="white", width=lw)
    d.arc([215, 300, 297, 345], 0, 180, fill="white", width=lw)
    return img


def make_face_blurred(size=512, blur=5):
    return make_scribble_face_outline(size).filter(ImageFilter.GaussianBlur(blur))


def make_face_boxed(size=512, blur=5, lw=3):
    img = Image.new("RGB", (size, size), "black")
    d = ImageDraw.Draw(img)
    d.ellipse([140, 100, 372, 410], outline="white", width=lw)
    d.ellipse([185, 205, 225, 228], outline="white", width=lw)
    d.ellipse([287, 205, 327, 228], outline="white", width=lw)
    d.line([256, 240, 248, 295], fill="white", width=lw)
    d.line([248, 295, 264, 295], fill="white", width=lw)
    d.arc([218, 315, 294, 355], 0, 180, fill="white", width=lw)
    d.rectangle([140, 100, 372, 410], outline="white", width=lw)
    return img.filter(ImageFilter.GaussianBlur(blur))


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    face_blurred = make_face_blurred()
    face_blurred.save(os.path.join(out_dir, "face_blurred_scribble.png"))
    print(f"Saved: {os.path.join(out_dir, 'face_blurred_scribble.png')}")

    face_boxed = make_face_boxed()
    face_boxed.save(os.path.join(out_dir, "face_boxed_scribble.png"))
    print(f"Saved: {os.path.join(out_dir, 'face_boxed_scribble.png')}")
