# verl-apertus visual reasoning experiments

Apertus and VeRL glue code for four visual reasoning experiments on Clariden:

- **Lens search**: tool augmented knowledge VQA. The model calls a web image search
  tool (Google Cloud Vision, or cached results) to identify entities in an image and
  answer knowledge questions. Trained with SFT and GRPO.
- **Visual math reasoning**: free form mathematical problem solving from images. The
  model writes a step by step solution and a final answer, scored by an LLM judge.
  Trained with GRPO, no tools.
- **Line drawing**: the model calls a tool that draws line segments on the image to
  reason about spatial relationships. Two settings, a synthetic dot connecting task and
  a filtered ToolVQA subset. Trained with GRPO.
- **Rotate and flip**: the model calls a tool that rotates or flips the image to read
  content that is sideways, upside down, or mirrored. Synthetic OCR task plus a filtered
  ToolVQA subset. Trained with GRPO.

Each tool experiment ships a with tool run and a tool free ablation, plus direct Apertus
and Qwen baselines.

## Environment

Work from the scratch checkout:

```bash
cd ~/capscratch/verl-apertus      # = /capstor/scratch/cscs/$USER/verl-apertus
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl
```

SLURM jobs assume these sibling paths:

```text
~/capscratch/verl-apertus   # this repo (configs, data prep, tools, rewards, eval)
~/capscratch/verl           # VeRL checkout (the trainer)
~/capscratch/Emu3.5/src     # Emu3.5 vision tokenizer source (IBQ image tokens)
```

Most jobs run in the Clariden container:

```text
#SBATCH --environment=verl_env
```

### VeRL checkout

Training is run by the sibling VeRL checkout, not by a trainer copied into this repo.
The SLURM scripts call:

```bash
python3 -m verl.trainer.main_ppo
```

from `~/capscratch/verl`. If that checkout has moved, restore the working revision before
launching:

```bash
cd ~/capscratch/verl
git checkout 2815eea5d6a5153594dcd71637c423984a268791
```

All RL runs start from the shared Apertus image SFT base checkpoint:

```text
/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF
```

## Repository layout

```text
configs/      VeRL GRPO and SFT configs, tool schemas, model and agent loop configs
data_prep/    dataset download, filtering, and IBQ image encoding into parquet
tools/        tool implementations called during rollouts
rewards/      custom reward functions wired into VeRL
inference/    baselines, LLM judges, and plotting
evaluation/   accuracy computation from VeRL rollout logs
slurm/        Clariden job scripts for data prep, training, baselines, and inference
scripts/      checkpoint conversion helpers
```

Images are encoded to inline IBQ tokens by `inference/vision.py` using the Emu3.5 vision
tokenizer. `data_prep/apertus_rl_dataset.py` provides the Apertus tool agent loop, and
`configs/apertus_agent_loop.yaml` registers the loops used at rollout time.

## Lens search

The model answers a knowledge question about an image. When the answer needs outside
knowledge it calls `lens_search_tool`, which runs Google Cloud Vision web detection (or
returns cached FVQA results), then calls `display_answers` once with the final answer.

Data comes from FVQA. The first rows build the SFT set with cached search results, the
remaining rows build the pure RL set.

```bash
sbatch slurm/prepare_lens_search.slurm          # build SFT and RL parquet
sbatch slurm/lens_search_sft.slurm              # SFT
sbatch slurm/lens_search_merge_sft.slurm        # merge FSDP shards to HF
sbatch slurm/lens_search_rl_armA.slurm          # arm A: pure RL from the base checkpoint
sbatch slurm/lens_search_rl_armB.slurm          # arm B: RL from the SFT checkpoint
```

Evaluation runs an out of distribution sweep over A-OKVQA, InfoSeek, and TextVQA, scored
by an LLM judge:

```bash
sbatch slurm/run_lens_ood_eval.slurm
```

Key files:

```text
data_prep/lens_search_common.py
data_prep/prepare_lens_search_sft.py
data_prep/prepare_lens_search_sft_toolvqa.py
data_prep/prepare_lens_search_rl.py
data_prep/prepare_lens_search_rl_full_fvqa.py
configs/lens_search_sft.yaml
configs/lens_search_rl_grpo_armA.yaml
configs/lens_search_rl_grpo_armB.yaml
configs/lens_search_tool_config.yaml
tools/lens_search_tool.py
rewards/lens_search_rl_reward.py
inference/llm_judge_lens.py
inference/run_lens_ood_eval.py
inference/lens_search_rl_plots.py
scripts/merge_lens_sft_fsdp_to_hf.py
```

## Visual math reasoning

The model solves an image based math problem in free prose and writes the final answer
on the last line. No tools are used. The reward is an LLM judge (Qwen3-32B) that checks
semantic equivalence to the ground truth, served by a sidecar sglang endpoint.

Training data is drawn from Math-VR-train and DynaMath (variants 1 to 9), with DynaMath
variant 10 held out for in distribution evaluation. Out of distribution evaluation uses
MathVision and MathVerse.

```bash
sbatch slurm/prepare_visual_math_rl_data.slurm  # build train and eval parquet
sbatch slurm/qwen_judge_server.slurm            # start the Qwen3-32B judge endpoint
sbatch slurm/visual_math_rl_train.slurm         # GRPO, external judge endpoint
sbatch slurm/visual_math_rl_v3_2node.slurm      # GRPO, judge and trainer co-scheduled
```

Evaluation and plotting:

```bash
python inference/visual_math_rl_compare.py
python inference/visual_math_rl_plots.py
```

Key files:

```text
data_prep/visual_math_rl_common.py
data_prep/prepare_visual_math_rl_mathvr.py
data_prep/prepare_visual_math_rl_dynamath.py
data_prep/prepare_visual_math_rl_eval.py
configs/visual_math_rl_grpo.yaml
rewards/visual_math_rl_reward.py
inference/llm_judge_math.py
```

## Line drawing

`line_drawing_tool` draws line segments between points on the image, then returns the
annotated image as IBQ tokens. The model uses it to connect objects, align locations, or
mark spatial relationships before answering.

Synthetic task: identify which labeled target (A, B, C, or D) lies on the straight line
between a red dot and a blue dot.

```bash
python data_prep/prepare_line_drawing_rl_generate.py   # generate images and jsonl
sbatch slurm/prepare_line_drawing_rl.slurm             # encode and build parquet
sbatch slurm/line_drawing_rl.slurm --fresh             # GRPO with the tool
sbatch slurm/line_drawing_rl_no_tool.slurm --fresh     # tool free ablation
```

ToolVQA task: text answer ToolVQA rows filtered to spatial questions where a line helps.

```bash
DOWNLOAD_TOOLVQA=1 sbatch slurm/prepare_toolvqa_rl.slurm
sbatch slurm/toolvqa_rl.slurm --fresh
```

Baselines, inference sweeps, and plotting:

```bash
sbatch slurm/run_line_drawing_apertus_baseline.slurm
sbatch slurm/run_line_drawing_qwen_baseline.slurm
STEPS="10 20 30 40 50 60 70" sbatch slurm/run_line_drawing_rl_inference_sweep.slurm
python evaluation/plot_line_drawing_rl_curve.py
```

Key files:

```text
data_prep/prepare_line_drawing_rl_generate.py
data_prep/prepare_line_drawing_rl_parse.py
data_prep/prepare_toolvqa_download.py
data_prep/prepare_toolvqa_rl_parse.py
configs/line_drawing_rl_grpo.yaml
configs/line_drawing_rl_no_tool_grpo.yaml
configs/line_drawing_rl_tool_config.yaml
configs/toolvqa_rl_grpo.yaml
configs/toolvqa_rl_tool_config.yaml
tools/line_drawing_emu_tool.py
evaluation/compute_line_drawing_rl_accuracy.py
evaluation/compute_toolvqa_rl_accuracy.py
evaluation/plot_line_drawing_rl_curve.py
```

## Rotate and flip

`rotate_flip_tool` applies rotations (30 to 330 degrees) and flips (horizontal, vertical,
transpose, transverse) to the image and returns the result as IBQ tokens. Up to four
operations can be chained. The model uses it when image content is sideways, upside down,
or mirrored.

Synthetic task: read short text rendered on an image after a random rotation or flip has
been applied.

```bash
sbatch slurm/prepare_rotate_flip_rl.slurm              # generate, encode, build parquet
sbatch slurm/rotate_flip_rl.slurm --fresh              # GRPO with the tool
sbatch slurm/rotate_flip_rl_no_tool.slurm --fresh      # tool free ablation
```

ToolVQA task: text answer ToolVQA rows filtered to OCR heavy questions. The shared parser
`data_prep/prepare_toolvqa_rl_parse.py` wires the tool selected with `--tool-name`:

```bash
DOWNLOAD_TOOLVQA=1 sbatch slurm/prepare_rotate_flip_toolvqa_rl.slurm
sbatch slurm/rotate_flip_toolvqa_rl.slurm --fresh
```

Baselines, inference sweeps, and plotting:

```bash
sbatch slurm/run_rotate_flip_apertus_baseline.slurm
sbatch slurm/run_rotate_flip_qwen_baseline.slurm
sbatch slurm/run_rotate_flip_rl_inference_sweep.slurm
python evaluation/plot_rotate_flip_rl_curve.py
```

Key files:

```text
data_prep/prepare_rotate_flip_rl_generate.py
data_prep/prepare_rotate_flip_rl_parse.py
configs/rotate_flip_rl_grpo.yaml
configs/rotate_flip_rl_no_tool_grpo.yaml
configs/rotate_flip_rl_tool_config.yaml
configs/rotate_flip_toolvqa_rl_grpo.yaml
configs/rotate_flip_toolvqa_rl_tool_config.yaml
tools/rotate_flip_emu_tool.py
evaluation/compute_rotate_flip_rl_accuracy.py
evaluation/plot_rotate_flip_rl_curve.py
```

## Notes

- `rewards/cof_rl_reward.py` and `data_prep/prepare_cof_rl_parse.py` keep their original
  names from earlier work. The synthetic line drawing and rotate flip configs reuse the
  exact match reward, and the lens search data prep reuses shared constants from the
  parser, so both modules are kept.
- The synthetic tasks score with exact final answer matching. Lens search and visual math
  score with an LLM judge.
- Tool free ablations use the `single_turn_agent` loop and remove all tool kwargs, so they
  share the same examples as the with tool runs.
