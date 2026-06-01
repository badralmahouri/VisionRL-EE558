"""Generate synthetic line-drawing RL data.

The default behavior is intentionally non-destructive: if raw.jsonl already
exists, it is kept exactly as-is so the existing train/val split remains stable.
A held-out test_raw.jsonl is generated beside it when missing.

Examples:
    python data_prep/prepare_line_drawing_rl_generate.py
    python data_prep/prepare_line_drawing_rl_generate.py --num-test-samples 500
    python data_prep/prepare_line_drawing_rl_generate.py --force-test
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

QUESTION = "Which labeled target, A, B, C, or D, lies on the straight line connecting the centers of the red and blue dots?"
LETTERS = ("A", "B", "C", "D")
CANVAS_SIZE = 512
MARGIN = 52
DOT_RADIUS = 10
TARGET_RADIUS = 8
MIN_ENDPOINT_DISTANCE = 220
MIN_TARGET_SEPARATION = 58
MAX_LINE_DISTANCE = 9.0


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_line_distance(point: tuple[int, int], start: tuple[int, int], end: tuple[int, int]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = math.hypot(dx, dy)
    if denom == 0:
        return distance(point, start)
    return abs(dy * px - dx * py + ex * sy - ey * sx) / denom


def interpolate(start: tuple[int, int], end: tuple[int, int], t: float) -> tuple[int, int]:
    return (round(start[0] + (end[0] - start[0]) * t), round(start[1] + (end[1] - start[1]) * t))


def random_point(rng: random.Random) -> tuple[int, int]:
    return (rng.randint(MARGIN, CANVAS_SIZE - MARGIN), rng.randint(MARGIN, CANVAS_SIZE - MARGIN))


def sample_endpoints(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(10_000):
        red = random_point(rng)
        blue = random_point(rng)
        if distance(red, blue) >= MIN_ENDPOINT_DISTANCE:
            return red, blue
    raise RuntimeError("failed to sample endpoints")


def valid_distractor(candidate: tuple[int, int], red: tuple[int, int], blue: tuple[int, int], points: list[tuple[int, int]]) -> bool:
    if point_line_distance(candidate, red, blue) <= MAX_LINE_DISTANCE * 2.5:
        return False
    if min(distance(candidate, point) for point in [red, blue, *points]) < MIN_TARGET_SEPARATION:
        return False
    return True


def sample_record(question_id: int, image_file: str, rng: random.Random) -> dict[str, Any]:
    red, blue = sample_endpoints(rng)
    answer = rng.choice(LETTERS)

    # Put the correct target clearly between the colored dots, away from endpoints.
    target_on_line = interpolate(red, blue, rng.uniform(0.28, 0.72))
    targets: dict[str, tuple[int, int]] = {answer: target_on_line}

    distractors: list[tuple[int, int]] = [target_on_line]
    for letter in LETTERS:
        if letter == answer:
            continue
        for _ in range(10_000):
            candidate = random_point(rng)
            if valid_distractor(candidate, red, blue, distractors):
                targets[letter] = candidate
                distractors.append(candidate)
                break
        else:
            raise RuntimeError(f"failed to sample distractor for {letter}")

    return {
        "question_id": question_id,
        "image_file": image_file,
        "question": QUESTION,
        "answer": answer,
        "points": {
            "red": list(red),
            "blue": list(blue),
            "targets": {letter: list(targets[letter]) for letter in LETTERS},
        },
    }


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_dot(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def draw_record(record: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)

    for value in range(64, CANVAS_SIZE, 64):
        draw.line((value, 0, value, CANVAS_SIZE), fill=(238, 238, 238), width=1)
        draw.line((0, value, CANVAS_SIZE, value), fill=(238, 238, 238), width=1)

    points = record["points"]
    red = tuple(points["red"])
    blue = tuple(points["blue"])

    draw_dot(draw, red, DOT_RADIUS, (220, 40, 35), (130, 0, 0))
    draw_dot(draw, blue, DOT_RADIUS, (30, 85, 220), (0, 25, 120))

    font = load_font(24)
    for letter in LETTERS:
        point = tuple(points["targets"][letter])
        draw_dot(draw, point, TARGET_RADIUS, (250, 250, 250), (35, 35, 35))
        label_pos = (point[0] + 13, point[1] - 15)
        draw.text(label_pos, letter, fill=(25, 25, 25), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


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
        draw_record(record, output_dir / image_file)
        records.append(record)
    write_jsonl(output_dir / raw_name, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic line-drawing RL raw data")
    parser.add_argument("--output-dir", default="data_prep/line_drawing_rl")
    parser.add_argument("--num-samples", type=int, default=1000, help="train/val raw sample count")
    parser.add_argument("--num-test-samples", type=int, default=200, help="held-out test sample count")
    parser.add_argument("--seed", type=int, default=42, help="train/val generation seed")
    parser.add_argument("--test-seed", type=int, default=4242, help="test generation seed")
    parser.add_argument("--force-trainval", action="store_true", help="regenerate raw.jsonl and line_*.png")
    parser.add_argument("--force-test", action="store_true", help="regenerate test_raw.jsonl and test_line_*.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw.jsonl"
    test_path = output_dir / "test_raw.jsonl"

    if raw_path.exists() and not args.force_trainval:
        print(f"Keeping existing train/val raw data: {raw_path}")
    else:
        print(f"Generating {args.num_samples} train/val raw examples -> {raw_path}")
        generate_split(output_dir, "raw.jsonl", "line", args.num_samples, args.seed)

    if test_path.exists() and not args.force_test:
        print(f"Keeping existing test raw data: {test_path}")
    else:
        print(f"Generating {args.num_test_samples} test raw examples -> {test_path}")
        generate_split(output_dir, "test_raw.jsonl", "test_line", args.num_test_samples, args.test_seed)

    print("Done.")


if __name__ == "__main__":
    main()
