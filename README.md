# GlimpseBC

Code for **GlimpseBC: Learning Saccadic Visual Policies from Global View Demonstrations**.
The repository supports training GlimpseBC and baselines on Robomimic, PushT, and LIBERO-Spatial. 

## Setup

Clone the repository and install system libraries:

```bash
git clone https://github.com/ijay-narang/GlimpseBC
cd GlimpseBC
bash scripts/install_rendering_deps.sh
```

This installs OSMesa for CPU rendering and OpenGL/EGL libraries.

### Robomimic

Create and activate the environment:

```bash
uv sync
source .venv/bin/activate
```

Download the datasets:

```bash
bash scripts/robomimic/utils/robomimic_fetch_data.sh
```

Train GlimpseBC on Lift:

```bash
python scripts/train.py \
  --device cuda \
  --benchmark robomimic \
  --task lift \
  --method glimpse \
  --policy chiunet \
  --crop-size 32
```

### PushT

Use the same environment as Robomimic. If it is not already created and active:

```bash
uv sync
source .venv/bin/activate
```

Download the dataset into the Hugging Face cache used by training:

```bash
bash scripts/pusht/utils/pusht_fetch_data.sh
```

Training extracts the cached archive automatically. Train GlimpseBC:

```bash
python scripts/train.py \
  --device cuda \
  --benchmark pusht \
  --method glimpse \
  --policy chiunet \
  --crop-size 32
```

### LIBERO-Spatial

Create and activate the separate LIBERO environment:

```bash
bash scripts/libero/utils/setup_libero_env.sh
source .venv-libero/bin/activate
```

Download the ten LIBERO-Spatial tasks:

```bash
bash scripts/libero/utils/libero_fetch_data.sh
```

Train GlimpseBC on task 0:

```bash
python scripts/train.py \
  --device cuda \
  --benchmark libero \
  --task 0 \
  --method glimpse \
  --policy rnn \
  --crop-size 32
```

### GLiTR

The GLiTR baseline adapts [GliTr: Glimpse Transformers With Spatiotemporal Consistency for Online Action Prediction](https://openaccess.thecvf.com/content/WACV2023/html/Rangrej_GliTr_Glimpse_Transformers_With_Spatiotemporal_Consistency_for_Online_Action_Prediction_WACV_2023_paper.html)
by Rangrej et al. (WACV 2023) to behavioral cloning. It uses the
[official implementation](https://github.com/facebookresearch/GliTr) and requires
a separate setup. Initialize the upstream model and download its pretrained iBOT weights:

```bash
git submodule update --init --recursive
bash glitr/download_ibot_checkpoint.sh
```

Train one full-frame teacher per task. For Robomimic Can:

```bash
python scripts/train_glitr.py \
  --device cuda \
  --benchmark robomimic --task can \
  --stage teacher \
  --output-dir run_logs/can_teacher
```

After the teacher finishes, train a student using its exported checkpoint:

```bash
python scripts/train_glitr.py \
  --device cuda \
  --benchmark robomimic --task can \
  --stage student --crop-size 32 \
  --teacher-checkpoint run_logs/can_teacher/teacher.pt
```

For PushT, use `--benchmark pusht` and omit `--task`. For LIBERO, use
`--benchmark libero --task 0` (or another task ID) in the LIBERO environment.

GliTr uses its own architecture, so it has no `--policy` option.
Weights default to `glitr/pretrained` and can be overriden with `--pretrained-dir` or
`GLITR_PRETRAINED_DIR`.

### Training options

Both launchers accept `--device cuda` (default) or `--device cpu`. Replace
`--device cuda` in the examples above to run on CPU. CPU training is slower;
use `--steps 10` for a short smoke run (evaluation still runs at the end).
For annealed smoke runs, also reduce `--anneal-steps` to at most `--steps`.

CPU mode defaults to OSMesa software rendering for Robomimic and LIBERO, which
requires the system OSMesa library. GPU mode defaults to EGL. Existing
`MUJOCO_GL` and `PYOPENGL_PLATFORM` environment settings override these defaults.
PushT does not use MuJoCo.

Each command starts one run immediately. Training seed is fixed to **0** and
evaluation seed to **10000**. Both launchers default to **200,000 updates** and
**batch size 32**; change these with `--steps` and `--batch-size`.

For `scripts/train.py`:

| Option | Choices |
| --- | --- |
| `--method` | `full`, `center`, `glimpse`, `annealed`, `foveated` |
| `--policy` | `rnn` (regression), `chiunet` (flow matching) |
| `--crop-size` | `32`, `48`, `64`; also PushT `24` or LIBERO `96`; omit for `full` |
| `--start-size` | Required for `annealed`; initial source-region width in pixels |
| `--anneal-steps` | Annealed only; defaults to `100000`, must not exceed `--steps` |

`center` uses a fixed center crop; `glimpse` learns fixations; `annealed` shrinks
the source region during training while keeping the output crop size fixed.
`foveated` uses a circular view with a sharp center and blurred periphery and is
available for PushT and LIBERO. These options expose supported implementations;
not every combination is an experiment reported in the paper.

For example, a full-frame baseline and an annealed PushT run:

```bash
python scripts/train.py --benchmark pusht --method full --policy rnn
python scripts/train.py --benchmark pusht --method annealed --policy rnn \
  --crop-size 32 --start-size 96
```

The full-frame PushT baseline retains an 84x84 random crop of its 96x96 source;
Robomimic and LIBERO full-frame baselines use the full image.
Evaluation runs at the final update: 100 episodes for Robomimic/PushT and 50 for
LIBERO. ChiUNet is evaluated with 9/3/1 sampling steps; RNN uses one step.

Logs, configuration, and checkpoints are saved in a new timestamped directory
under `run_logs/`. Use `--output-dir` to choose a new directory. Existing output
is not automatically resumed. GLiTR stage logs are stored beneath its output
directory. Output directories are created automatically.

## Attribution

This repository builds on the [behavioral-cloning implementation](https://github.com/simchowitzlabpublic/much-ado-about-noising)
of [Much Ado About Noising: Dispelling the Myths of Generative Robotic Control](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html)
by Pan et al. (ICLR 2026). The GLiTR baseline uses the
[official GliTr implementation](https://github.com/facebookresearch/GliTr) by
Rangrej et al. (WACV 2023), pinned in `.gitmodules`. See [LICENSE](LICENSE)
and the upstream projects for licensing and attribution.

## References

1. Chaoyi Pan, Giridharan Anantharaman, Nai-Chieh Huang, Claire Jin, Daniel Pfrommer,
   Chenyang Yuan, Frank Permenter, Guannan Qu, Nicholas Boffi, Guanya Shi, and Max Simchowitz.
   [Much Ado About Noising: Dispelling the Myths of Generative Robotic Control](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html).
   *International Conference on Learning Representations (ICLR)*, 2026.
2. Samrudhdhi B. Rangrej, Kevin J. Liang, Tal Hassner, and James J. Clark.
   [GliTr: Glimpse Transformers With Spatiotemporal Consistency for Online Action Prediction](https://openaccess.thecvf.com/content/WACV2023/html/Rangrej_GliTr_Glimpse_Transformers_With_Spatiotemporal_Consistency_for_Online_Action_Prediction_WACV_2023_paper.html).
   *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*,
   pages 3413-3423, 2023.

### BibTeX

```bibtex
@inproceedings{pan2026much,
  title={Much Ado About Noising: Dispelling the Myths of Generative Robotic Control},
  author={Pan, Chaoyi and Anantharaman, Giridharan and Huang, Nai-Chieh and Jin, Claire and Pfrommer, Daniel and Yuan, Chenyang and Permenter, Frank and Qu, Guannan and Boffi, Nicholas and Shi, Guanya and Simchowitz, Max},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://proceedings.iclr.cc/paper_files/paper/2026/hash/9299ca9ed58731945e934adb5b71728c-Abstract-Conference.html}
}

@inproceedings{rangrej2023glitr,
  title={{GliTr}: Glimpse Transformers With Spatiotemporal Consistency for Online Action Prediction},
  author={Rangrej, Samrudhdhi B. and Liang, Kevin J. and Hassner, Tal and Clark, James J.},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={3413--3423},
  year={2023},
  url={https://openaccess.thecvf.com/content/WACV2023/html/Rangrej_GliTr_Glimpse_Transformers_With_Spatiotemporal_Consistency_for_Online_Action_Prediction_WACV_2023_paper.html}
}
```
