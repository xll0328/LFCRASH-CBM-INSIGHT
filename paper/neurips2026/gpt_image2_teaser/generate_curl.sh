#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../../.." && pwd)"

prompt_path="${1:-${script_dir}/prompt_v2.md}"
model="${AIHUBMIX_IMAGE_MODEL:-gpt-image-2}"
size="${AIHUBMIX_IMAGE_SIZE:-1536x1024}"
quality="${AIHUBMIX_IMAGE_QUALITY:-high}"
n="${AIHUBMIX_IMAGE_N:-1}"
base_url="${AIHUBMIX_BASE_URL:-https://api.aihubmix.com/v1}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_model="${model//[^A-Za-z0-9_.-]/_}"
out_dir="${AIHUBMIX_IMAGE_OUT_DIR:-${script_dir}/outputs/${timestamp}_curl_${safe_model}_${quality}}"

mkdir -p "${out_dir}"

python3 - "${prompt_path}" "${out_dir}/request.json" "${model}" "${size}" "${quality}" "${n}" <<'PY'
import json
import pathlib
import sys

prompt_path, request_path, model, size, quality, n = sys.argv[1:]
prompt = pathlib.Path(prompt_path).read_text(encoding="utf-8")
payload = {
    "model": model,
    "prompt": prompt,
    "n": int(n),
    "size": size,
    "quality": quality,
}
pathlib.Path(request_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

if [[ -z "${AIHUBMIX_API_KEY:-}" ]]; then
  printf 'AIHUBMIX_API_KEY: ' >&2
  stty -echo
  IFS= read -r AIHUBMIX_API_KEY
  stty echo
  printf '\n' >&2
fi

env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  curl -sS --fail-with-body --max-time 900 --connect-timeout 60 --http1.1 --noproxy '*' \
  -H "Authorization: Bearer ${AIHUBMIX_API_KEY}" \
  -H 'Content-Type: application/json' \
  "${base_url%/}/images/generations" \
  --data-binary @"${out_dir}/request.json" \
  > "${out_dir}/response.json"

python3 - "${out_dir}" "${root_dir}" "${prompt_path}" "${model}" "${size}" "${quality}" <<'PY'
import base64
import json
import pathlib
import sys
import urllib.request

out_dir = pathlib.Path(sys.argv[1])
root_dir = pathlib.Path(sys.argv[2])
prompt_path = pathlib.Path(sys.argv[3])
model, size, quality = sys.argv[4:]
response_path = out_dir / "response.json"
payload = json.loads(response_path.read_text(encoding="utf-8"))

written = []
for idx, item in enumerate(payload.get("data", [])):
    if not isinstance(item, dict):
        continue
    image_path = out_dir / f"insight_teaser_{idx:02d}.png"
    if item.get("b64_json"):
        image_path.write_bytes(base64.b64decode(item["b64_json"]))
        item["b64_json"] = "<omitted>"
        written.append(str(image_path.relative_to(root_dir)))
    elif item.get("url"):
        with urllib.request.urlopen(item["url"], timeout=120) as response:
            image_path.write_bytes(response.read())
        item["url"] = "<downloaded>"
        written.append(str(image_path.relative_to(root_dir)))

manifest = {
    "mode": "curl-openai-direct",
    "endpoint": "https://api.aihubmix.com/v1/images/generations",
    "prompt": str(prompt_path.relative_to(root_dir)) if prompt_path.is_relative_to(root_dir) else str(prompt_path),
    "model": model,
    "size": size,
    "quality": quality,
    "outputs": written,
    "raw_response_without_key": payload,
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps({"manifest": str((out_dir / "manifest.json").relative_to(root_dir)), "outputs": written}, indent=2))
if not written:
    raise SystemExit("No image data found in response")
PY
