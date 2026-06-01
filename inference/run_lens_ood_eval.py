"""Standalone OOD eval for the Apertus lens-search agent (no training).

Runs multi-turn inference (turn 1: lens_search call, turn 2: display_answers)
on the 200-row OOD benchmarks (aokvqa / textvqa / infoseek) and writes a JSONL
compatible with inference/score_lens_baseline_regex.py and inference/llm_judge_lens.py.

Pure inference. Mirrors the verl Apertus tool_agent_loop but standalone —
no Ray, no DDP, no actor/ref load. Reads an already-merged HF checkpoint and
calls LensSearchTool (live by default; cached supported for testing).

Defense-in-depth against runaway GCP cost:
  --gcp-call-limit N   HARD CAP on total tool calls per run (default 30 = smoke)
                       Each call = 1 GCP Vision annotate_image() requesting
                       3 features (WEB + LANDMARK + TEXT). Aborts the row that
                       would exceed the cap.

Usage:
    # 5-row smoke (SFT-only ckpt, live)
    python inference/run_lens_ood_eval.py \\
        --ckpt /capstor/scratch/cscs/$USER/verl-apertus/checkpoints/apertus8b-lens-search-sft-toolvqa/global_step_390_hf \\
        --bench aokvqa \\
        --limit 5 \\
        --gcp-call-limit 30 \\
        --tool-mode live

    # Full 200-row eval (cap at 1000 calls; expect ~250)
    python inference/run_lens_ood_eval.py \\
        --ckpt <merged_global_step_40> \\
        --bench infoseek \\
        --gcp-call-limit 1000 \\
        --tool-mode live
"""
from __future__ import annotations

import sys
import os

# CRITICAL: Emu3.5/src has an empty proto/__init__.py that shadows proto-plus
# when EMU3_SRC is on PYTHONPATH. google-cloud-vision needs proto-plus's
# `proto.module(...)` API and will fail with AttributeError. Force proto-plus
# to win the initial `import proto` by removing EMU3_SRC from sys.path for the
# duration of that import, then restoring it for the later vision_tokenizer import.
_EMU3_SHADOW_DIR = "/users/badralmahouri/Emu3.5/src"
_emu3_was_present = _EMU3_SHADOW_DIR in sys.path
if _emu3_was_present:
    sys.path.remove(_EMU3_SHADOW_DIR)
import proto  # noqa: E402 — must precede any indirect import via google-cloud-vision
assert hasattr(proto, "module"), (
    f"proto-plus shadowing fix failed: proto resolves to {proto.__file__}"
)
if _emu3_was_present:
    sys.path.append(_EMU3_SHADOW_DIR)  # restore for `from vision_tokenizer import ...`

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


APERTUS_SYSTEM = "You are a helpful assistant with access to tools."

APERTUS_INSTRUCTION = (
    "If you need information about the entity, landmark, object, or text in the "
    "image, call the lens_search tool with a short natural-language query. Use "
    "the search results to inform your answer. Then call the display_answers "
    "tool exactly once at the end of your response, passing your final answer "
    "as the single element of the `answers` argument."
)


def find_json_arrays(text: str) -> list[str]:
    """Balanced-bracket scanner — mirrors rewards/lens_search_rl_reward.py."""
    if not text:
        return []
    arrays, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                arrays.append(text[start : i + 1])
                start = None
    return arrays


def extract_first_lens_search_call(text: str) -> dict | None:
    """Return the first {"lens_search": {...}} dict in the text, or None."""
    for arr in find_json_arrays(text):
        try:
            calls = json.loads(arr)
        except json.JSONDecodeError:
            continue
        if not isinstance(calls, list):
            continue
        for c in calls:
            if isinstance(c, dict) and "lens_search" in c and isinstance(c["lens_search"], dict):
                return c["lens_search"]
    return None


def has_display_answers(text: str) -> bool:
    for arr in find_json_arrays(text):
        try:
            calls = json.loads(arr)
        except json.JSONDecodeError:
            continue
        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict) and "display_answers" in c:
                    return True
    return False


def load_manifest(bench: str) -> list[dict]:
    path = PROJECT_ROOT / "data_prep" / "lens_baseline" / bench / "manifest.json"
    with open(path) as f:
        rows = json.load(f)
    # Skip rows whose image is not on disk or status != ok
    return [r for r in rows if r.get("status") == "ok" and r.get("png") and os.path.exists(r["png"])]


def make_lens_tool(mode: str):
    """Instantiate LensSearchTool in cached or live mode."""
    from tools.lens_search_tool import LensSearchTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
    schema = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "lens_search",
            "description": "Search the web for information about an entity, landmark, object, or text visible in the user's image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short natural-language query."},
                },
                "required": ["query"],
            },
        },
    )
    return LensSearchTool(config={"type": "native", "mode": mode, "max_response_chars": 1800}, tool_schema=schema)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF-merged checkpoint dir (config.json + safetensors).")
    ap.add_argument("--bench", required=True, choices=["aokvqa", "textvqa", "infoseek"])
    ap.add_argument("--output", default=None, help="Output JSONL path (default auto under evaluation/lens_ood_outputs/).")
    ap.add_argument("--limit", type=int, default=None, help="If set, eval only the first N rows.")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Per-turn cap.")
    ap.add_argument("--tool-mode", default="live", choices=["live", "cached"],
                    help="LensSearchTool mode. 'cached' won't work for OOD images (no cache); use for testing only.")
    ap.add_argument("--gcp-call-limit", type=int, default=30,
                    help="HARD CAP on total tool calls per run (defense vs. runaway cost). Aborts the row that would exceed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.tool_mode == "live":
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set; required for --tool-mode live", file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
            print(f"ERROR: GOOGLE_APPLICATION_CREDENTIALS file not found: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}", file=sys.stderr)
            sys.exit(2)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_label = Path(args.ckpt).name.replace("/", "_")
        ts = int(time.time())
        out_dir = PROJECT_ROOT / "evaluation" / "lens_ood_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"lens_ood_{args.bench}_{ckpt_label}_{args.tool_mode}_{ts}.jsonl"

    # Lazy heavy imports
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from PIL import Image
    from inference.vision import encode_image, load_vq_model
    from data_prep.lens_search_common import LENS_SEARCH_TOOL

    DISPLAY_ANSWERS_TOOL = {
        "name": "display_answers",
        "description": "Display the final answer to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {"type": "array", "items": {"type": "string"}, "description": "The final answer(s)."},
            },
            "required": ["answers"],
        },
    }

    print(f"[{time.strftime('%H:%M:%S')}] Loading manifest for bench={args.bench} ...", flush=True)
    rows = load_manifest(args.bench)
    print(f"  manifest rows: {len(rows)} (status=ok and image-on-disk)", flush=True)
    if args.limit:
        rows = rows[: args.limit]
        print(f"  limited to first {len(rows)} rows", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading VQ model + tokenizer + model ...", flush=True)
    # IBQ Vision Tokenizer (same path used in configs/apertus.yaml and slurm/run_baseline.slurm)
    vq_model_path = os.environ.get(
        "VQ_MODEL_PATH",
        "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer",
    )
    if not os.path.isdir(vq_model_path):
        print(f"ERROR: VQ model dir not found: {vq_model_path}", file=sys.stderr)
        sys.exit(2)
    print(f"  vq_model_path={vq_model_path}")
    vq_model = load_vq_model(vq_model_path, device="cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda:0").eval()
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    # Stop generation at </s>, <|assistant_end|>, OR <|tools_suffix|>.
    # The Apertus SFT model reliably emits <|tools_suffix|> after each tool
    # call (it's the closing delimiter of [{"...": {...}}]) but does NOT
    # reliably emit <|assistant_end|> — so stopping only at </s>/<|assistant_end|>
    # lets generation run to max_new_tokens and produce loopy garbage.
    # Stopping at <|tools_suffix|> ends each turn cleanly right after the tool call.
    stop_token_ids = [tokenizer.eos_token_id]
    for tok in ("<|assistant_end|>", "<|tools_suffix|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_token_ids.append(tid)
    print(f"  stop_token_ids={stop_token_ids}  "
          f"(eos={tokenizer.eos_token_id}, "
          f"assistant_end={tokenizer.convert_tokens_to_ids('<|assistant_end|>')}, "
          f"tools_suffix={tokenizer.convert_tokens_to_ids('<|tools_suffix|>')})", flush=True)

    # Instantiate the lens tool
    print(f"[{time.strftime('%H:%M:%S')}] Initializing LensSearchTool (mode={args.tool_mode}) ...", flush=True)
    tool = make_lens_tool(args.tool_mode)

    # Run loop
    gcp_calls = 0
    n_lens_called = 0
    n_display_answers = 0
    results = []

    for i, row in enumerate(rows):
        t_row = time.time()
        question = row["question"]
        image_path = row["png"]
        # Encode image to IBQ tokens
        try:
            image = Image.open(image_path).convert("RGB")
            image_tok_str = encode_image(image, vq_model)
        except Exception as e:
            print(f"  [skip {i}] image load/encode failed: {e}", flush=True)
            continue

        user_content = f"{image_tok_str}\n{question}\n\n{APERTUS_INSTRUCTION}"
        messages = [
            {"role": "system", "content": APERTUS_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=[LENS_SEARCH_TOOL, DISPLAY_ANSWERS_TOOL],
            enable_thinking=True,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

        # ---- Turn 1: model decides whether to call lens_search ----
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        turn1_ids = output_ids[0, input_ids.shape[1]:]
        turn1_text = tokenizer.decode(turn1_ids, skip_special_tokens=False)

        full_gen = turn1_text
        lens_query = None
        lens_result = None
        lens_error = None

        ls_call = extract_first_lens_search_call(turn1_text)
        if ls_call is not None:
            lens_query = ls_call.get("query")
            # ---- GCP HARD CAP ----
            if args.tool_mode == "live" and gcp_calls + 1 > args.gcp_call_limit:
                lens_error = (
                    f"GCP_CALL_LIMIT_REACHED ({args.gcp_call_limit}); aborting tool call. "
                    f"Row counted as no-tool."
                )
                print(f"  [row {i}] {lens_error}", flush=True)
            else:
                # Execute the tool synchronously via asyncio
                async def run_tool():
                    instance_id, _ = await tool.create(create_kwargs={"image_path": image_path, "data_id": row["name"]})
                    return await tool.execute(instance_id, {"query": lens_query})

                try:
                    response, _, info = asyncio.run(run_tool())
                    if args.tool_mode == "live":
                        gcp_calls += 1
                    n_lens_called += 1
                    lens_result = response.text if response else ""
                except Exception as e:
                    lens_error = f"tool exception: {type(e).__name__}: {e}"
                    print(f"  [row {i}] {lens_error}", flush=True)
                    lens_result = f"Error: {lens_error}"

            if lens_result is not None:
                # Append inline `[result]` (matches Apertus jinja2 tool-msg rendering),
                # then continue generation in the same assistant turn.
                appended = output_ids[0].tolist() + tokenizer.encode(
                    f"[{lens_result}]", add_special_tokens=False
                )
                input_ids2 = torch.tensor([appended], device="cuda:0")
                with torch.no_grad():
                    output_ids2 = model.generate(
                        input_ids2,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=stop_token_ids,
                    )
                turn2_ids = output_ids2[0, input_ids2.shape[1]:]
                turn2_text = tokenizer.decode(turn2_ids, skip_special_tokens=False)
                full_gen = turn1_text + f"[{lens_result}]" + turn2_text

        if has_display_answers(full_gen):
            n_display_answers += 1

        gold_answers = row.get("gold_answers") or [row.get("answer", "")]
        rec = {
            # fields used by score_lens_baseline_regex.py + llm_judge_lens.py
            "name": row["name"],
            "model_output": full_gen,
            "gold_answers": gold_answers,
            "lens_capability": row.get("lens_capability"),
            # extras for our own analysis
            "bench": args.bench,
            "question": question,
            "lens_called": ls_call is not None and lens_result is not None and lens_error is None,
            "lens_query": lens_query,
            "lens_error": lens_error,
            "has_display_answers": has_display_answers(full_gen),
            "elapsed_s": round(time.time() - t_row, 1),
            "gcp_calls_so_far": gcp_calls,
        }
        results.append(rec)
        print(
            f"  [{i+1}/{len(rows)}] {row['name']} | "
            f"lens={'Y' if rec['lens_called'] else 'N'} | "
            f"display={'Y' if rec['has_display_answers'] else 'N'} | "
            f"gcp={gcp_calls} | t={rec['elapsed_s']}s | "
            f"q={(lens_query or '')[:60]!r}",
            flush=True,
        )

    # Dump
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print()
    print("=" * 60)
    print(f"OOD eval done — bench={args.bench}, ckpt={Path(args.ckpt).name}, mode={args.tool_mode}")
    print("=" * 60)
    print(f"  rows processed:           {len(results)}")
    print(f"  lens_search executed:     {n_lens_called}/{len(results)} ({100*n_lens_called/max(1,len(results)):.1f}%)")
    print(f"  display_answers emitted:  {n_display_answers}/{len(results)} ({100*n_display_answers/max(1,len(results)):.1f}%)")
    print(f"  total GCP calls used:     {gcp_calls} (cap was {args.gcp_call_limit})")
    print(f"  output written to:        {out_path}")
    print()
    print("Next step: score with inference/score_lens_baseline_regex.py --input " + str(out_path))


if __name__ == "__main__":
    main()
