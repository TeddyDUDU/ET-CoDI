# ET-CoDI

Event-triggered cluster management with shared-state reuse and StreamHAC, for scheduling streaming inference at the edge.

## Environment

Python 3.10.

```bash
pip install -r requirements.txt
```

The CPU simulation uses `requirements.txt`. Packages used only by the Stable Diffusion 3 run are listed under GPU experiment. They are not in `requirements.txt`.

## Training the checkpoints

`experiments/exp1_train_ippo.py` and `experiments/exp3_train_cm.py` implement the paper's training protocol. The checkpoints are not distributed with the repository. Runtime and hardware follow the script defaults.

```bash
python experiments/exp1_train_ippo.py
```

This writes `results/exp1_ippo/trained_ippo.pt`. The script creates that directory.

```bash
python experiments/exp3_train_cm.py
```

This writes `results/exp3_train_cm/trained_cm.pt`. It loads the IPPO checkpoint from exp1 and creates its output directory.

## Reproducing the paper

Run exp1 and exp3 first to generate the checkpoints. exp2 needs only `results/exp1_ippo/trained_ippo.pt`. exp5, exp6, exp7, exp10, exp12, and exp14 need that file and `results/exp3_train_cm/trained_cm.pt`.

These commands write under `results/`. The repository does not ship those outputs. A full run is required before `--plot-only` on the scripts that accept it (`exp5`, `exp12`, `exp14`); that flag only redraws from files the full run already wrote.

| Paper item | Command | Output |
| --- | --- | --- |
| Table main | `python experiments/exp6_threshold.py` | `results/exp6_threshold/summary.txt` |
| Fig. exp1 | `python experiments/exp2_drift_degradation.py` | `results/exp2_drift/D_t_pub_vs_t.png`, `Q_vs_t.png`, `U_vs_t.png` |
| Fig. exp5 | `python experiments/exp5_sensitivity.py` | `results/exp5_sensitivity/utility_bars_grouped.png` |
| Table ablation / Fig. comove | `python experiments/exp7_refresh.py` | `results/exp7_refresh/summary.txt`, `comove_S.png` |
| Fig. up | `python experiments/exp10_drift_factors.py` | `results/exp10_drift_factors/U_vs_p_drift.png`, `U_vs_sigma.png` |
| Fig. scale | `python experiments/exp12_system.py` | `results/exp12_system/latency_vs_bandwidth.png`, `makespan_vs_users.png` |
| Fig. real | `python experiments/exp14_controlled.py` | `results/exp14_controlled/` (see below) |

Fig. real needs a local Stable Diffusion 3 Medium tree and a GPU. Pass the tree with `SD3_MODEL_DIR` or `--model-path`:

```bash
python experiments/exp14_controlled.py --model-path "$SD3_MODEL_DIR"
```

exp14 needs a GPU environment and SD3 weights prepared by the user.

## Other experiments

`run_all.py` runs exp1, exp2, exp3, exp5, exp6, exp7, and exp10. It trains the IPPO checkpoint in exp1, then runs exp2, which needs that IPPO checkpoint and does not need the CM checkpoint. exp3 then trains the CM checkpoint, and the later steps evaluate with both files. It does not run exp12 or exp14.

```bash
python run_all.py
python run_all.py --only exp6
python run_all.py --dry-run
```

## Results

Each experiment creates its `results/exp*/` directory and writes a `summary.txt`.

## GPU experiment

exp14 is the only GPU experiment in this tree. It calls the shared Stable Diffusion 3 helpers in `experiments/sd3_utils.py`. Those helpers import `torch`, `diffusers`, `transformers`, `PIL`, and `ImageReward`. `torch` is already in `requirements.txt`.

```bash
pip install diffusers transformers image-reward pillow
```

Additional packages may be required by diffusers, transformers, or ImageReward: accelerate, safetensors, sentencepiece, protobuf, ftfy, timm. Install them if you encounter missing-module errors.

`PIL` is the `pillow` distribution, and `ImageReward` is the `image-reward` distribution. Set `SD3_MODEL_DIR` or pass `--model-path`. exp14 also needs a local CLIP model for scoring. Set --clip-model or CLIP_MODEL_DIR to a local CLIP model directory; otherwise scoring will fail with an explicit error. This repository does not include a Slurm script. Image and sheet outputs are written under `results/exp14_controlled/`, including `images/`, `sheets/`, `exp14_reward_delta.png`, and `exp14_examples.png`.

## Layout

```text
algorithms/    cluster manager, StreamHAC, cache, shared IPPO
baselines/     cluster and published comparison policies
core/          parameters, paths, utility, IPPO, clustering
env/           static and streaming environments, drift
experiments/   experiment entry points, including exp14 and sd3_utils.py
results/       empty until a run; generated summaries and figures
```

## Citation and license

Fangbo Ni, Liying Yang, Jianxiong Guo, Jiandian Zeng, Zhiqing Tang, Tian Wang, and Weijia Jia. "ET-CoDI: Event-Triggered Shared-State Management for Collaborative Diffusion Inference." Manuscript submitted to IEEE Transactions on Cloud Computing (under review).

MIT License (see LICENSE).
