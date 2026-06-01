"""Merge the Arm B Lens-search SFT FSDP shards into a single HF directory.

Wraps verl's `FSDPModelMerger` with a small monkey-patch that fixes a bug in
its `_load_and_merge_state_dicts` method: when a tensor is REPLICATED across
ranks (i.e., not a DTensor — typically buffers like rotary cos/sin caches or
scalar config tensors), the upstream code at fsdp_model_merger.py:202 tries
to `torch.cat([t, t, t, t], dim=0)`, which both (a) duplicates the tensor
4× incorrectly and (b) crashes on zero-dim tensors with
``RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated``.

The fix: when the key is NOT in `param_placements` (= replicated), just take
the first shard's copy.

CLI mirrors `python -m verl.model_merger merge`:
    python scripts/merge_lens_sft_fsdp_to_hf.py \\
        --local_dir <repo>/checkpoints/apertus8b-lens-search-sft/global_step_16 \\
        --target_dir <repo>/checkpoints/apertus8b-lens-search-sft/global_step_16_hf
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm


def _patched_load_and_merge_state_dicts(
    self,
    world_size: int,
    total_shards: int,
    mesh_shape: tuple,
    mesh_dim_names: tuple,
) -> dict:
    """Drop-in replacement for FSDPModelMerger._load_and_merge_state_dicts.

    Same logic as upstream verl, except the `else` branch (non-DTensor key —
    tensor is replicated across ranks) takes shard 0 instead of concatenating
    4 identical copies.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed._tensor import Shard  # noqa: F401  (typing)

    model_state_dict_lst = [None] * total_shards

    def process_one_shard(rank: int, lst: list):
        model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
        lst[rank] = sd
        return sd

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
        for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
            future.result()

    state_dict: dict = {}
    param_placements: dict[str, tuple] = {}

    for key in set(model_state_dict_lst[0].keys()):
        state_dict[key] = []
        for model_state_shard in model_state_dict_lst:
            tensor = model_state_shard.pop(key)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    n_sharded = 0
    n_replicated = 0
    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            continue
        if key in param_placements:
            placements = param_placements[key]
            if len(mesh_shape) == 1:
                assert len(placements) == 1
                state_dict[key] = self._merge_by_placement(state_dict[key], placements[0])
                n_sharded += 1
            else:
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            # PATCH: replicated tensor — take shard 0 instead of cat (which
            # crashes on zero-dim tensors and duplicates 4× for non-zero-dim).
            state_dict[key] = state_dict[key][0]
            n_replicated += 1

    print(f"  Merged: {n_sharded} sharded + {n_replicated} replicated keys")
    return state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="FSDP shards dir (contains model_world_size_N_rank_K.pt + huggingface/ subdir)")
    ap.add_argument("--target_dir", required=True, help="Output HF dir")
    ap.add_argument("--backend", default="fsdp")
    args = ap.parse_args()

    # Apply the monkey-patch BEFORE importing FSDPModelMerger so the fixed
    # method is in place when the class is constructed.
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    FSDPModelMerger._load_and_merge_state_dicts = _patched_load_and_merge_state_dicts

    os.makedirs(args.target_dir, exist_ok=True)

    # Match `generate_config_from_args` (verl/model_merger/base_model_merger.py:127-159):
    # explicit `<local_dir>/huggingface` as hf_model_config_path (BaseModelMerger
    # passes this directly to AutoConfig.from_pretrained); trust_remote_code=True
    # is required for Apertus (custom HF model code).
    cfg = ModelMergerConfig(
        operation="merge",
        backend=args.backend,
        local_dir=args.local_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
        target_dir=args.target_dir,
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=True,
        is_value_model=False,
        use_cpu_initialization=False,
    )
    print(f"Config: {cfg}")
    merger = FSDPModelMerger(cfg)
    merger.merge_and_save()
    if hasattr(merger, "cleanup"):
        merger.cleanup()
    print(f"\nDone. Output written to: {args.target_dir}")


if __name__ == "__main__":
    main()
