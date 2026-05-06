#!/usr/bin/env python3
"""Generate INSIGHT teaser candidates with GPT Image 2 through AIHubMix.

The API key is intentionally read only from the environment. Do not hard-code
keys in this file or pass them on the command line.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True, help="Prompt markdown file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--model", default=os.environ.get("AIHUBMIX_IMAGE_MODEL", "gpt-image-2"))
    parser.add_argument("--model-path", default=os.environ.get("AIHUBMIX_IMAGE_MODEL_PATH", "openai/gpt-image-2"))
    parser.add_argument("--mode", choices=["openai", "predictions"], default="openai")
    parser.add_argument("--base-url", default=os.environ.get("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"))
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--size", default="1536x1024")
    parser.add_argument("--quality", default="high")
    return parser.parse_args()


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return json.loads(response.model_dump_json())


def strip_large_fields(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(payload))
    for item in cleaned.get("data", []):
        if "b64_json" in item:
            item["b64_json"] = "<omitted>"
    return cleaned


def decode_data_url(value: str) -> bytes | None:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", value)
    if not match:
        return None
    return base64.b64decode(match.group(1))


def save_output_value(value: Any, image_path: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False

    if decoded := decode_data_url(value):
        image_path.write_bytes(decoded)
        return True

    if re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", value) and len(value) > 2000:
        try:
            image_path.write_bytes(base64.b64decode(value))
            return True
        except Exception:
            pass

    if value.startswith("http://") or value.startswith("https://"):
        with urllib.request.urlopen(value, timeout=120) as response:
            image_path.write_bytes(response.read())
        return True

    return False


def extract_prediction_outputs(payload: dict[str, Any]) -> list[Any]:
    for key in ("output", "data", "images"):
        value = payload.get(key)
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                found: list[Any] = []
                for item in value:
                    for field in ("b64_json", "url", "image", "image_url", "base64", "base64_json"):
                        if field in item:
                            found.append(item[field])
                return found
            return value
    nested = payload.get("prediction")
    if isinstance(nested, dict):
        return extract_prediction_outputs(nested)
    return []


def call_openai_images(args: argparse.Namespace, api_key: str, prompt: str, out_dir: Path) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    response = client.images.generate(
        model=args.model,
        prompt=prompt,
        n=args.n,
        size=args.size,
        quality=args.quality,
    )
    payload = response_to_dict(response)

    written: list[str] = []
    for index, item in enumerate(response.data):
        b64_json = getattr(item, "b64_json", None)
        if not b64_json:
            continue
        image_path = out_dir / f"insight_teaser_{index:02d}.png"
        image_path.write_bytes(base64.b64decode(b64_json))
        written.append(str(image_path.relative_to(ROOT)))
    return {"payload": payload, "written": written}


def call_predictions(args: argparse.Namespace, api_key: str, prompt: str, out_dir: Path) -> dict[str, Any]:
    url = f"{args.base_url.rstrip('/')}/models/{args.model_path}/predictions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "input": {
                "prompt": prompt,
                "size": args.size,
                "n": args.n,
                "quality": args.quality,
                "output_format": "png",
            }
        },
        timeout=300,
    )
    payload = response.json()
    if response.status_code >= 400:
        raise SystemExit(f"Prediction request failed: HTTP {response.status_code}: {payload}")

    written: list[str] = []
    for index, value in enumerate(extract_prediction_outputs(payload)):
        image_path = out_dir / f"insight_teaser_{index:02d}.png"
        if save_output_value(value, image_path):
            written.append(str(image_path.relative_to(ROOT)))
    return {"payload": payload, "written": written}


def main() -> int:
    args = parse_args()
    prompt_path = args.prompt.resolve()
    api_key = os.environ.get("AIHUBMIX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing AIHUBMIX_API_KEY. Set it in the environment; do not write it into files."
        )

    prompt = prompt_path.read_text(encoding="utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "openai":
        result = call_openai_images(args, api_key, prompt, out_dir)
    else:
        result = call_predictions(args, api_key, prompt, out_dir)

    payload = result["payload"]
    written = result["written"]

    manifest = {
        "created_at": timestamp,
        "model": args.model,
        "model_path": args.model_path,
        "mode": args.mode,
        "base_url": args.base_url,
        "size": args.size,
        "quality": args.quality,
        "n": args.n,
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "outputs": written,
        "raw_response_without_key": strip_large_fields(payload),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "outputs": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
