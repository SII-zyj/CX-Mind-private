<div align="center">
  <img src="logo.png" alt="CX-Mind logo" height="100">
  <h1 style="font-size: 32px; font-weight: bold;">CX-Mind: Curriculum-Guided Multimodal Reasoning for Chest X-rays</h1>

  <br>

  <a href="https://arxiv.org/abs/2508.03733">
    <img src="https://img.shields.io/badge/ArXiv-CXMind-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/SII-JasperLi77/CX-Mind">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-purple" alt="checkpoint">
  </a>
</div>

## CX-Mind

CX-Mind is a multimodal large language model (MLLM) for diagnostic reasoning on chest X-rays. The project is built on a highly modular reinforcement learning (RL) training stack that combines Ray, FSDP, and vLLM to deliver large-scale distributed self-alignment. The core idea is to use curriculum-style rewards to guide the model toward the "think–answer–critique" pattern across interleaved image-text reasoning chains, improving both radiology-specific and open-ended QA accuracy.

**Key insights:**

- **Distributed PPO training pipeline**: `verl.trainer` orchestrates multi-role workers (Actor/Rollout/Ref/Critic) across Ray remote processes, injecting reward functions and data pipelines at runtime to provide the end-to-end RL training loop.【F:verl/trainer/main.py†L17-L124】
- **Unified image/video data interface**: `RLHFDataset` supports text-only, single/multi-image, and video samples, automatically rescaling pixels, filling templates, and filtering prompts to respect multimodal constraints.【F:verl/utils/dataset.py†L54-L214】
- **Highly configurable training parameters**: All hyperparameters (data, algorithm, worker, trainer) are managed through the `PPOConfig` dataclass and interoperate seamlessly with YAML configuration files.【F:verl/trainer/config.py†L34-L178】【F:examples/config.yaml†L1-L103】
- **Pluggable rewards and custom metrics**: Reward modules support batched or sequential execution and can load local Python functions that encode domain-specific scoring (e.g., joint F1 and format consistency).【F:verl/workers/reward/config.py†L20-L38】【F:examples/reward_function/openQA.py†L6-L102】

## Quick Start

### Environment Setup

1. Prepare a machine that satisfies your GPU/CPU budget; Python 3.9+ is recommended (the default ruff profile targets Py39).【F:pyproject.toml†L18-L39】
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```
3. Configure API keys for services like Weights & Biases or Hugging Face before launching training.

### Start Training

The following example shows how to bootstrap a Ray cluster and launch GRPO training with the sample configuration.

```bash
# Stop any previous Ray cluster running on the local machine
ray stop

# Start a single-node Ray cluster (tune GPU count and port as needed)
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --port 8262

# Point to a local checkpoint cloned under ./checkpoints/
MODEL_PATH=./checkpoints/Qwen2.5-VL-7B-Instruct

# Launch training with inline overrides
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8
```

During execution, `verl.trainer.main` parses configuration overrides, initializes dataloaders and reward functions, and dispatches training jobs to remote FSDP workers.【F:verl/trainer/main.py†L34-L94】

## Programming Guide

### General Introduction

The CX-Mind training topology combines a Ray control plane with multiple worker roles:

- **Actor / Ref / Critic workers**: Derived from `FSDPWorker`, they load models, build FSDP/ULysses hybrid parallelism, and execute PPO updates or reference forward passes.【F:verl/workers/fsdp_workers.py†L33-L174】
- **Rollout engine**: Uses the vLLM inference backend to generate high-throughput samples with tunable parameters (temperature, top-p, tensor parallelism) and supports evaluation-specific sampling strategies.【F:verl/workers/rollout/config.py†L22-L48】
- **Ray worker group & `ResourcePoolManager`**: Automatically allocates GPU resources so Actor/Rollout/Ref/Critic roles can scale elastically while sharing a unified pool.【F:verl/trainer/main.py†L52-L93】

This design decouples policy optimization, experience collection, reward computation, and weight synchronization, making it easy to mix curriculum rewards with multimodal datasets.

### Training Config

All hyperparameters use a dataclass + YAML configuration flow:

- **`DataConfig`**: Controls training/validation datasets, field names, modality transforms, prompt length, and pixel bounds. Relative paths are resolved to absolute paths at load time.【F:verl/trainer/config.py†L34-L71】
- **`AlgorithmConfig`**: Defines PPO/GRPO knobs like KL penalties, advantage estimators, and online filtering windows.【F:verl/trainer/config.py†L74-L103】
- **`WorkerConfig`**: Splits actor, reference, critic, rollout, and reward modules, each with model loading, optimizer, FSDP, and offloading settings.【F:verl/workers/config.py†L31-L50】【F:examples/config.yaml†L34-L85】
- **`TrainerConfig`**: Specifies training epochs, logging, evaluation/saving cadence, and multi-node GPU topology.【F:verl/trainer/config.py†L106-L155】【F:examples/config.yaml†L86-L103】

You can load a base configuration via `python -m verl.trainer.main config=path/to.yaml` and override individual keys on the command line for quick ablation studies.【F:verl/trainer/main.py†L97-L108】

### Use your own data

`RLHFDataset` expects data in the following format:

- Datasets can be local JSON/JSONL files, directories, or Hugging Face Hub datasets, optionally with `split` suffixes (for example, `dataset@train`).【F:verl/utils/dataset.py†L93-L139】
- Default fields include `prompt`, `answer`, `images`, and `videos`. The loader automatically applies pixel rescaling, RGB normalization, and chat-template formatting when multimodal inputs are provided.【F:verl/utils/dataset.py†L153-L210】
- Set `filter_overlong_prompts` to drop samples that exceed maximum token length during loading, preventing mid-training interruptions.【F:verl/utils/dataset.py†L146-L214】

A minimal JSONL entry with a single image looks like this:

```jsonl
{"prompt": "<image>Please describe the pathology", "answer": "<think>…reasoning…</think><answer>Atelectasis</answer>", "images": ["samples/sample_001.png"]}
```

If image paths are relative, configure `data.image_dir` to point to the base folder. Video inputs can be supplied through the `videos` field with `video_fps` set in the config.

### Implement your own tools

- **Custom rewards**: Point `worker.reward.reward_function=./path/to/file.py:func` in the config to load a Python callable. The framework selects batch or sequential execution based on `reward_type` and injects dependencies such as tokenizers automatically.【F:verl/workers/reward/config.py†L20-L38】【F:verl/trainer/main.py†L68-L78】
- **Reward example**: `examples/reward_function/openQA.py` combines `<think>/<answer>` tag validation with multi-label F1 scoring, returning multiple metrics for filtering and logging.【F:examples/reward_function/openQA.py†L6-L102】
- **Policy extensions**: To swap the rollout engine or add new worker roles, extend the base classes under `verl/workers/rollout` or `verl/workers/fsdp_workers.py` and follow the Ray dispatch interface to integrate your component.【F:verl/workers/fsdp_workers.py†L91-L174】【F:verl/workers/rollout/config.py†L22-L48】

**Important:**

- Run the primary training job on worker nodes instead of the Ray head node to avoid scheduling conflicts.【F:verl/trainer/main.py†L29-L94】
- Large models demand significant GPU memory and storage bandwidth; consider enabling parameter/offload options (`offload_params`, `offload_optimizer`) based on your hardware profile.【F:examples/config.yaml†L56-L80】
- Mixed precision and Ulysses padding-free optimizations are enabled by default—tune `fsdp` and `padding_free` settings if your hardware requires alternative configurations.【F:verl/workers/actor/config.py†L41-L75】

## Star Chart

[![Star History Chart](https://api.star-history.com/svg?repos=SII-JasperLi77/CX-Mind&type=Date)](https://star-history.com/#SII-JasperLi77/CX-Mind&Date)

## License

This project is released under the [Apache License](./LICENSE).

## Citation

If CX-Mind is helpful for your research or product, please cite our paper:

```bibtex
@article{CXMind2025,
  title={CX-Mind: Curriculum-Guided Multimodal Reasoning for Chest X-rays},
  author={Li, Jasper and others},
  journal={arXiv preprint arXiv:2508.03733},
  year={2025}
}
```
