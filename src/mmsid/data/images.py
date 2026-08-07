"""Download one product image per item, resize it and record which items have one.

Images are saved to $RS_DATA/raw/images/<name>/<item_idx>.jpg with the short side at 256 px.
They are only used to compute embeddings and are never committed.

    python -m mmsid.data.images --name video_games
"""

import argparse
import asyncio
import io
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mmsid.paths import processed_dir, raw_dir

CONCURRENCY = 32
RETRIES = 3
SHORT_SIDE = 256
DELAY = 0.05  # seconds each worker waits after a request


def image_dir(name: str) -> Path:
    return raw_dir() / "images" / name


def resize_short_side(img: Image.Image, short: int = SHORT_SIDE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = short / min(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return img


async def fetch_one(session: aiohttp.ClientSession, url: str, dest: Path) -> bool:
    for attempt in range(RETRIES):
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return False
                resp.raise_for_status()
                data = await resp.read()
            resize_short_side(Image.open(io.BytesIO(data))).save(dest, "JPEG", quality=90)
            return True
        except Exception:
            await asyncio.sleep(1.5 * (attempt + 1))
        finally:
            await asyncio.sleep(DELAY)
    return False


async def download_all(jobs: list[tuple[str, Path]]) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=30)
    bar = tqdm(total=len(jobs), desc="images", mininterval=5)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def worker(url: str, dest: Path) -> None:
            async with sem:
                await fetch_one(session, url, dest)
                bar.update()

        await asyncio.gather(*(worker(u, d) for u, d in jobs))
    bar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    items_path = processed_dir(args.name) / "items.parquet"
    items = pd.read_parquet(items_path)
    out = image_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (url, out / f"{idx}.jpg")
        for idx, url in zip(items["item_idx"], items["image_url"], strict=True)
        if isinstance(url, str) and url and not (out / f"{idx}.jpg").exists()
    ]
    print(f"{len(jobs):,} images to fetch, {items['image_url'].notna().sum():,} items have a URL")
    if jobs:
        asyncio.run(download_all(jobs))

    items["has_image"] = [(out / f"{i}.jpg").exists() for i in items["item_idx"]]
    items.to_parquet(items_path, index=False)
    print(f"has_image: {items['has_image'].mean():.1%} of {len(items):,} items")


if __name__ == "__main__":
    main()
