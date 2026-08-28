"""Gradio demo: recommend from a history, explore the code space, and tokenise a new product.

    uv run python scripts/export_demo_bundle.py     # once, after training
    uv run python demo/app.py

The app reads everything from demo/bundle (override with MMSID_BUNDLE). Product images are loaded
from Amazon's URLs at display time. It runs on CPU.
"""

import html
import io
import json
import os
import pickle
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from mmsid.data.dataset import pad_histories, trim
from mmsid.models.tiger import Tiger
from mmsid.models.trie import PrefixIndex

BUNDLE = Path(os.environ.get("MMSID_BUNDLE", Path(__file__).parent / "bundle"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = {"T": "Text-only IDs (TIGER)", "LF": "Text + image IDs (late fusion)"}
TEXT_MODEL = "sentence-transformers/sentence-t5-base"
IMAGE_MODEL = "google/siglip-base-patch16-224"
MAX_HISTORY = 20

CSS = """
.cards {display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;}
.card {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px;
       background: var(--background-fill-secondary); font-size: 12px; line-height: 1.3;}
.card img {width: 100%; height: 130px; object-fit: contain; background: #fff; border-radius: 6px;}
.card .title {margin-top: 6px; height: 3.9em; overflow: hidden;}
.card .code {margin-top: 4px; font-family: monospace; color: var(--color-accent);}
.card .extra {color: var(--body-text-color-subdued);}
.cards.small {grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));}
.cards.small .card img {height: 90px;}
.cards.small .card .title {height: 2.6em;}
"""

meta = json.loads((BUNDLE / "meta.json").read_text())
items = pd.read_parquet(BUNDLE / "items.parquet").set_index("item_idx")
titles_lower = items["title"].str.lower()
text_emb = np.load(BUNDLE / "text.npy").astype(np.float32)
image_emb = np.load(BUNDLE / "image.npy").astype(np.float32)

IDS, MODELS, INDEXES = {}, {}, {}
for variant, info in meta["variants"].items():
    IDS[variant] = np.load(BUNDLE / f"ids_{variant}.npy")
    model = Tiger(IDS[variant], info["level_sizes"], **info["tiger"])
    model.load_state_dict(torch.load(BUNDLE / f"tiger_{variant}.pt", map_location="cpu"))
    MODELS[variant] = model.to(DEVICE).eval()
    INDEXES[variant] = PrefixIndex(IDS[variant], info["level_sizes"], DEVICE)


def variant_of(label: str) -> str:
    return next(k for k, v in LABELS.items() if v == label)


def code_str(variant: str, item: int) -> str:
    return "(" + ", ".join(str(c) for c in IDS[variant][item]) + ")"


def cards(entries, small: bool = False) -> str:
    """entries: (item_idx, code text, extra text). Returns an HTML grid of product cards."""
    out = []
    for item, code, extra in entries:
        row = items.loc[item]
        out.append(
            f'<div class="card"><img src="{html.escape(row["image_url"] or "")}" loading="lazy">'
            f'<div class="title">{html.escape(row["title"][:80])}</div>'
            + (f'<div class="code">{html.escape(code)}</div>' if code else "")
            + (f'<div class="extra">{html.escape(extra)}</div>' if extra else "")
            + "</div>")
    return f'<div class="cards{" small" if small else ""}">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- tab 1: recommend

def search(query: str):
    query = (query or "").strip().lower()
    if len(query) < 2:
        return gr.update(choices=[], value=None)
    hits = titles_lower[titles_lower.str.contains(query, regex=False)].index[:30]
    return gr.update(choices=[(items.loc[i, "title"][:90], int(i)) for i in hits], value=None)


def add_to_history(item, history):
    if item is not None and item not in history:
        history = (history + [int(item)])[-MAX_HISTORY:]
    return history, cards([(i, "", "") for i in history], small=True)


def clear_history():
    return [], "", "", ""


@torch.no_grad()
def recommend(history, label):
    if not history:
        return "", "Add at least one product to the history first."
    variant = variant_of(label)
    hist = trim(torch.from_numpy(pad_histories([history]))).to(DEVICE)
    user = torch.zeros(1, dtype=torch.long, device=DEVICE)  # a new user gets the first user token
    found, scores = MODELS[variant].recommend(user, hist, INDEXES[variant], beam=50)
    top = [(int(i), f"ID {code_str(variant, int(i))}", f"log p {s:.2f}")
           for i, s in zip(found[0, :10].tolist(), scores[0, :10].tolist(), strict=True) if i > 0]
    return cards(top), f"Top 10 of 50 beams, {LABELS[variant]}."


# ---------------------------------------------------------------- tab 2: code explorer

def second_tokens(label, c1):
    ids = IDS[variant_of(label)][1:]
    counts = Counter(ids[ids[:, 0] == int(c1), 1].tolist())
    choices = [(f"{c}  ({n} products)", c) for c, n in sorted(counts.items(), key=lambda x: -x[1])]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def explore(label, c1, c2):
    if c1 is None or c2 is None:
        return "", ""
    variant = variant_of(label)
    ids = IDS[variant]
    members = np.flatnonzero((ids[:, 0] == int(c1)) & (ids[:, 1] == int(c2)))
    members = members[members > 0]
    cats = Counter(items.loc[members, "leaf_category"]).most_common(3)
    summary = (f"**{len(members)} products** share the prefix ({c1}, {c2}). Most common categories: "
               + ", ".join(f"{c or 'unknown'} ({n})" for c, n in cats))
    return cards([(int(i), code_str(variant, int(i)), "") for i in members[:48]], small=True), summary


# ---------------------------------------------------------------- tab 3: cold item

@lru_cache(maxsize=1)
def encoders():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor

    text = SentenceTransformer(TEXT_MODEL, device=DEVICE)
    siglip = AutoModel.from_pretrained(IMAGE_MODEL).to(DEVICE).eval()
    return text, siglip, AutoProcessor.from_pretrained(IMAGE_MODEL)


@lru_cache(maxsize=1)
def quantizers():
    with open(BUNDLE / "quantizers_LF.pkl", "rb") as f:
        fitted = pickle.load(f)
    for _, q in fitted:
        if hasattr(q, "model"):
            q.model.to(DEVICE)
            q.device = DEVICE
    return fitted


@torch.no_grad()
def cold_item(title: str, image_url: str):
    if not title or not image_url:
        return "Give both a title and an image URL.", ""
    text_model, siglip, processor = encoders()
    t = text_model.encode([title], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    img = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=20).read())).convert("RGB")
    pixels = processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)
    feats = siglip.get_image_features(pixel_values=pixels)
    feats = feats if torch.is_tensor(feats) else feats.pooler_output
    v = torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()

    inputs = {"text": t, "image": v}
    codes = np.concatenate([q.encode(inputs[m])[0] for m, q in quantizers()], axis=1)[0]
    ids = IDS["LF"]
    same_prefix = int(((ids[1:, 0] == codes[0]) & (ids[1:, 1] == codes[1])).sum())
    same_code = int((ids[1:, : len(codes)] == codes).all(1).sum())

    sim = text_emb[1:] @ t[0] + image_emb[1:] @ v[0]
    nearest = np.argsort(-sim)[:10] + 1
    shared = [int((ids[i, : len(codes)] == codes).cumprod().sum()) for i in nearest]
    text = (f"**Assigned content code: ({', '.join(map(str, codes))})**, two tokens from the title "
            f"and two from the image. A collision token would be added on top. {same_prefix} "
            f"catalogue products share its first two tokens and {same_code} share all four.\n\n"
            f"Nearest catalogue products by text and image similarity:")
    near = [(int(i), f"ID {code_str('LF', int(i))}", f"{n} leading tokens shared")
            for i, n in zip(nearest, shared, strict=True)]
    return text, cards(near)


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Multimodal semantic IDs") as app:
    gr.Markdown(f"# Multimodal semantic IDs: generative recommendation\n"
                f"Catalogue: Amazon Reviews 2023, `{meta['data']}`, {len(items):,} products.")

    with gr.Tab("Recommend"):
        history = gr.State([])
        with gr.Row():
            query = gr.Textbox(label="Search products", placeholder="e.g. zelda, controller", scale=2)
            results = gr.Dropdown(label="Pick a match to add it to the history", choices=[],
                                  interactive=True, scale=3)
            clear = gr.Button("Clear", scale=1)
        gr.Markdown("**History** (oldest first)")
        history_view = gr.HTML()
        with gr.Row():
            which = gr.Radio(list(LABELS.values()), value=LABELS["LF"], label="Semantic IDs built from")
            go = gr.Button("Recommend", variant="primary")
        note = gr.Markdown()
        recs = gr.HTML()
        query.change(search, query, results)
        results.input(add_to_history, [results, history], [history, history_view])
        clear.click(clear_history, None, [history, history_view, recs, note])
        go.click(recommend, [history, which], [recs, note])
        which.input(recommend, [history, which], [recs, note])

    with gr.Tab("Code explorer"):
        gr.Markdown("Pick the first two tokens of an ID and see which products share them.")
        with gr.Row():
            which2 = gr.Radio(list(LABELS.values()), value=LABELS["LF"], label="Semantic IDs built from")
            first = gr.Dropdown(list(range(meta["variants"]["LF"]["level_sizes"][0])), value=0,
                                label="First token")
            second = gr.Dropdown([], label="Second token")
        summary = gr.Markdown()
        grid = gr.HTML()
        first.change(second_tokens, [which2, first], second)
        which2.change(second_tokens, [which2, first], second)
        second.change(explore, [which2, first, second], [grid, summary])
        app.load(second_tokens, [which2, first], second)

    with gr.Tab("Cold item"):
        gr.Markdown("Describe a product the model has never seen. It gets a late-fusion code from "
                    "its title and image alone. The first use loads the two encoders.")
        with gr.Row():
            new_title = gr.Textbox(label="Title", scale=2)
            new_image = gr.Textbox(label="Image URL", scale=2)
            assign = gr.Button("Assign a code", variant="primary")
        assigned = gr.Markdown()
        neighbours = gr.HTML()
        assign.click(cold_item, [new_title, new_image], [assigned, neighbours])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), css=CSS)
