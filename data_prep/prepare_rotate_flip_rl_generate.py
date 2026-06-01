"""Generate synthetic rotate/flip OCR RL data.

Each sample renders a mixed-case, deliberately misspelled word on a clean sign,
applies a random rotation and/or flip, and asks the model to recover the
canonical correctly spelled word. For example, the sign text can be
"restOurant" while the expected answer is "restaurant".
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

QUESTION = "What correctly spelled word is represented by the sign text?"
CANVAS_SIZE = (640, 360)
SIGN_SIZE = (600, 190)
WORDS = (
    ("restOurant", "restaurant"),
    ("hOspitol", "hospital"),
    ("AirpOart", "airport"),
    ("LibRery", "library"),
    ("mUseam", "museum"),
    ("pharMacey", "pharmacy"),
    ("StaShion", "station"),
    ("baKary", "bakery"),
    ("theAtor", "theater"),
    ("gAllary", "gallery"),
    ("boOkstor", "bookstore"),
    ("unIversety", "university"),
    ("cLinick", "clinic"),
    ("caFetaria", "cafeteria"),
    ("marKett", "market"),
    ("hoTell", "hotel"),
)
TRANSFORMS = (
    ("rotate_30", ["rotate_330"]),
    ("rotate_45", ["rotate_315"]),
    ("rotate_60", ["rotate_300"]),
    ("rotate_90", ["rotate_270"]),
    ("rotate_120", ["rotate_240"]),
    ("rotate_135", ["rotate_225"]),
    ("rotate_150", ["rotate_210"]),
    ("rotate_180", ["rotate_180"]),
    ("rotate_210", ["rotate_150"]),
    ("rotate_225", ["rotate_135"]),
    ("rotate_240", ["rotate_120"]),
    ("rotate_270", ["rotate_90"]),
    ("rotate_300", ["rotate_60"]),
    ("rotate_315", ["rotate_45"]),
    ("rotate_330", ["rotate_30"]),
    ("flip_horizontal", ["flip_horizontal"]),
    ("flip_vertical", ["flip_vertical"]),
    ("rotate_45_flip_horizontal", ["flip_horizontal", "rotate_315"]),
    ("rotate_90_flip_horizontal", ["flip_horizontal", "rotate_270"]),
    ("rotate_135_flip_horizontal", ["flip_horizontal", "rotate_225"]),
    ("rotate_225_flip_horizontal", ["flip_horizontal", "rotate_135"]),
    ("rotate_270_flip_horizontal", ["flip_horizontal", "rotate_90"]),
    ("rotate_315_flip_horizontal", ["flip_horizontal", "rotate_45"]),
)


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/LiberationMono-Bold.ttf",
        "/users/qxie/miniconda3/envs/verl/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    raise FileNotFoundError("No scalable TTF font found for synthetic rotate/flip generation.")


def apply_operation(image: Image.Image, operation: str) -> Image.Image:
    if operation.startswith("rotate_"):
        angle = int(operation.removeprefix("rotate_"))
        return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(246, 247, 249))
    if operation == "flip_horizontal":
        return ImageOps.mirror(image)
    if operation == "flip_vertical":
        return ImageOps.flip(image)
    raise ValueError(operation)


def apply_transform(image: Image.Image, transform_name: str) -> Image.Image:
    transformed = image
    for part in transform_name.split("_flip_"):
        if part.startswith("rotate_"):
            transformed = apply_operation(transformed, part)
        elif part == "horizontal":
            transformed = apply_operation(transformed, "flip_horizontal")
        elif part == "vertical":
            transformed = apply_operation(transformed, "flip_vertical")
        elif part.startswith("flip_"):
            transformed = apply_operation(transformed, part)
    return transformed


def render_upright_word(word: str, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", SIGN_SIZE, (246, 247, 249))
    draw = ImageDraw.Draw(image)
    sign_w, sign_h = SIGN_SIZE
    draw.rounded_rectangle((4, 4, sign_w - 4, sign_h - 4), radius=18, fill=(255, 255, 255), outline=(35, 35, 35), width=5)

    font_size = rng.randint(88, 108)
    while font_size >= 58:
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), word, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= sign_w - 56 and text_h <= sign_h - 40:
            break
        font_size -= 2
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (sign_w - text_w) // 2 - bbox[0]
    y = (sign_h - text_h) // 2 - bbox[1] - 4
    draw.text((x, y), word, fill=(20, 20, 20), font=font)
    return image


def sample_record(question_id: int, image_file: str, rng: random.Random) -> dict[str, Any]:
    word, canonical_answer = rng.choice(WORDS)
    transform_name, inverse_operations = rng.choice(TRANSFORMS)
    return {
        "question_id": question_id,
        "image_file": image_file,
        "question": QUESTION,
        "answer": word,
        "canonical_answer": canonical_answer,
        "transform": transform_name,
        "inverse_operations": inverse_operations,
    }


def draw_record(record: dict[str, Any], path: Path, rng: random.Random) -> None:
    upright = render_upright_word(record["answer"], rng)
    transformed = apply_transform(upright, record["transform"])
    canvas = Image.new("RGB", CANVAS_SIZE, (246, 247, 249))
    transformed.thumbnail((CANVAS_SIZE[0] - 32, CANVAS_SIZE[1] - 32), Image.Resampling.BICUBIC)
    x = (CANVAS_SIZE[0] - transformed.width) // 2
    y = (CANVAS_SIZE[1] - transformed.height) // 2
    canvas.paste(transformed, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_split(output_dir: Path, raw_name: str, image_prefix: str, count: int, seed: int) -> None:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for idx in range(count):
        image_file = f"images/{image_prefix}_{idx:06d}.png"
        record = sample_record(idx, image_file, rng)
        draw_record(record, output_dir / image_file, rng)
        records.append(record)
    write_jsonl(output_dir / raw_name, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rotate/flip RL raw data")
    parser.add_argument("--output-dir", default="data_prep/rotate_flip_rl")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-test-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--test-seed", type=int, default=4343)
    parser.add_argument("--force-trainval", action="store_true")
    parser.add_argument("--force-test", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw.jsonl"
    test_path = output_dir / "test_raw.jsonl"
    if raw_path.exists() and not args.force_trainval:
        print(f"Keeping existing train/val raw data: {raw_path}")
    else:
        print(f"Generating {args.num_samples} train/val raw examples -> {raw_path}")
        generate_split(output_dir, "raw.jsonl", "rotate_flip", args.num_samples, args.seed)
    if test_path.exists() and not args.force_test:
        print(f"Keeping existing test raw data: {test_path}")
    else:
        print(f"Generating {args.num_test_samples} test raw examples -> {test_path}")
        generate_split(output_dir, "test_raw.jsonl", "test_rotate_flip", args.num_test_samples, args.test_seed)
    print("Done.")


if __name__ == "__main__":
    main()
