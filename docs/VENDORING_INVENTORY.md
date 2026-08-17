# Vendoring Inventory — Phase 0

**Status:** investigation only. No code was changed to produce this document.
**Date of investigation:** 2026-08-13.
**Workspace investigated:** `/home/hatem/final_project/SonicSight` (WSL2 Ubuntu),
reached from Windows as `\\wsl.localhost\Ubuntu\home\hatem\final_project\SonicSight`.

| repository | remote | branch | HEAD | working tree |
|---|---|---|---|---|
| `SonicSightBackend` | `https://github.com/K1ller-Whale/SonicSightBackend` | `main` (`== origin/main`) | `17016dd350ec93f6c5941a325632fceb2c8ee0ab` | clean except untracked `src/ckpt/` (weights, gitignored by `**/ckpt/*.pth`) |
| `multisensory` | `https://github.com/mysterI0s/multisensory` | `master` (`== origin/master`) | `eb75ea5ab416fce5d6edd0a76f4b4a3caf8de639` | clean |

Every measurement in this document was produced by running the command shown.
Anything not run is labelled **not measured**.

---

## 1. How the backend reaches outside its own tree (2a)

There is **no** `.gitmodules`, `setup.py`, `pyproject.toml`, `setup.cfg`,
`Dockerfile` or editable install anywhere in `SonicSightBackend`. The coupling is
entirely (a) `sys.path` manipulation plus filesystem-relative paths, and (b) one
network fetch. Full list:

### 1.1 Runtime server path escapes — the speech engine

| # | file:line | escape | effect |
|---|---|---|---|
| E1 | `src/engines/multisensory_engine.py:134-136` | `default_multisensory_root()` returns `Path(__file__).resolve().parents[3] / "multisensory"` | hard-codes the sibling checkout: `SonicSight/SonicSightBackend/src/engines/` → `SonicSight/multisensory` |
| E2 | `src/engines/multisensory_engine.py:160` | `os.environ["MULTISENSORY_ROOT"]` overrides E1 | environment variable naming a directory outside the tree |
| E3 | `src/engines/multisensory_engine.py:162` | existence probe `root/src/sep_video.py` | the FileNotFoundError that FR-024 tolerates |
| E4 | `src/engines/multisensory_engine.py:166-168` | checkpoint default `root/results/nets/sep/full/net.tf-160000`, overridable by `MULTISENSORY_CHECKPOINT` | weights live in the sibling repo |
| E5 | `src/engines/multisensory_engine.py:172-173` | `sys.path.insert(0, str(root/"src"))` | the sibling repo's `src/` is put on the import path at load time |
| E6 | `src/engines/multisensory_engine.py:180-181` | `import sep_params`, `import sep_video` | flat top-level imports resolved only because of E5 |

E1–E6 are the **whole** of the runtime coupling. `MULTISENSORY_ROOT` /
`MULTISENSORY_CHECKPOINT` are documented in `README.md:151-152`.

### 1.2 Non-runtime path escapes — the load suite

These do not affect the server, but they do affect the measurement harness, and
they escape to the *other two* siblings, not just `multisensory`.

| # | file:line | escape | reaches |
|---|---|---|---|
| E7 | `loadtest/paths.py:17` | `WORKSPACE_ROOT = dirname(BACKEND_ROOT)` | the workspace root |
| E8 | `loadtest/paths.py:18` | `NFR_TARGETS_YAML = WORKSPACE_ROOT/docs/nfr/nfr_targets.yaml` | the **`docs` repository** |
| E9 | `loadtest/paths.py:20-22` | `MOBILE_PROTO = WORKSPACE_ROOT/SonicSightMobile/app/src/main/proto/sonicsight.proto` | the **mobile repository** |
| E10 | `loadtest/nfr.py:25` | reads `paths.NFR_TARGETS_YAML` | docs repo |
| E11 | `loadtest/runmeta.py:41` | `git rev-parse` inside `WORKSPACE_ROOT/SonicSightMobile` | mobile repo |
| E12 | `loadtest/runmeta.py:62` | SHA-256 of `nfr_targets.yaml` into run metadata | docs repo |
| E13 | `loadtest/scenarios.py:818-825` | byte-compares backend proto against `MOBILE_PROTO` — this **is** NFR-COMPAT-001 | mobile repo |
| E14 | `loadtest/scenarios.py:919` | writes a report under `WORKSPACE_ROOT/docs/...` | docs repo |
| E15 | `loadtest/run_func001.sh:45` | `VideoPreprocessor().preprocess("../data/test_instruments.mp4", ...)` | a `data/` directory outside the backend |

### 1.3 Benign, in-tree `sys.path` bootstraps (not escapes)

`src/run_servers.py:9-11`, `replay_client.py:64-67`, `pixel_ab_harness.py:36-37`,
`loadtest/paths.py:24-26`, and every `tests/test_*.py:~8` insert
`<backend>/src` — all inside the tree. `src/config.py:118` sets
`CKPT_ROOT = <backend>/src/ckpt`, in-tree. No action needed.

### 1.4 The escape that is not a filesystem path

`src/models/__init__.py:62` and `:66` call
`torchvision.models.resnet18(pretrained)` with `pretrained = True`, then
overwrite the weights with `frame_best.pth`. The `pretrained=True` call still
performs a **network download** of the ImageNet ResNet-18 weights on first use.

Verified present in the host cache:

```
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth   46 830 571 bytes
sha256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

Consequence: **a container with no network and no pre-seeded `TORCH_HOME` cannot
load the `sonicsight` engine at all.** This must be provisioned like any other
checkpoint (Phase 1 §3c / UC-13). It is not a behaviour change to provision it;
it is already a hard dependency, just an invisible one.

---

## 2. What is actually reached at runtime (2b)

### 2.1 Method

Two independent passes, then diffed.

**Static.** AST walk of `multisensory/src`, seeded from exactly what
`MultisensoryEngine.load()` imports (`sep_params`, `sep_video`), following only
targets that resolve inside that directory, and separating module-level imports
from imports inside function/class bodies.

**Instrumented.** A real inference: `MultisensoryEngine().load()` against the
real `net.tf-160000` checkpoint on `/gpu:0`, then `eval_stream_window()` on one
full 46 352-sample window and 63 × 224×224×3 frames, with `builtins.__import__`
wrapped to record import order, and `sys.modules` filtered to the multisensory
tree afterwards. Frame and audio content are synthetic; the graph, the
checkpoint, the session and the forward pass are the real ones. This traces
imports — it measures nothing about quality or latency.

Result of the instrumented run (stderr):

```
Created device /job:localhost/replica:0/task:0/device:GPU:0 with 4078 MB memory:
  -> device: 0, name: NVIDIA GeForce GTX 1660 Ti, compute capability: 7.5
Loaded cuDNN version 91900
LOADED device=/gpu:0
INFERENCE OK  left=(46352,) right=(46352,) heatmap=(56, 56) conf=0.7908
```

### 2.2 Imported and executed — the vendoring set (14 files)

Order of first import, from the instrumented run:

`aolib.util`, `aolib.img`, `tfutil`, `sep_params`, `shift_params`, `shift_dset`,
`shift_net`, `aolib.sound`, `sep_dset`, `aolib.imtable`, `soundrep`, `sourcesep`,
`sep_video` — plus `aolib/__init__.py`, executed as the package initialiser.

| file | bytes |
|---|---|
| `src/aolib/__init__.py` | 191 |
| `src/aolib/img.py` | 9 070 |
| `src/aolib/imtable.py` | 24 515 |
| `src/aolib/sound.py` | 20 101 |
| `src/aolib/util.py` | 95 832 |
| `src/sep_dset.py` | 14 096 |
| `src/sep_params.py` | 4 459 |
| `src/sep_video.py` | 22 992 |
| `src/shift_dset.py` | 11 792 |
| `src/shift_net.py` | 27 068 |
| `src/shift_params.py` | 8 892 |
| `src/soundrep.py` | 3 307 |
| `src/sourcesep.py` | 40 874 |
| `src/tfutil.py` | 22 868 |
| **total** | **306 057 bytes (≈ 299 KiB)** |

`aolib/__init__.py` contains only an `__all__` list and imports nothing.

### 2.3 Imported but not executed

**None.** Every module in 2.2 is imported at module level and its top-level code
runs. There is no "imported but dormant" tier in this dependency set.

### 2.4 Present in the repo and never touched (7 files, 83 235 bytes)

| file | bytes | note |
|---|---|---|
| `src/aolib/areload.py` | 2 861 | dev reload helper |
| `src/aolib/subband.py` | 16 487 | lazily referenced by `aolib/sound.py` inside a function that is never called |
| `src/cam_analyze.py` | 21 302 | **CAM validation probe** — the reduction the engine reimplements (`multisensory_engine.py:74-87` cites `cam_analyze.reduce_positive`) |
| `src/gpu_doctor.py` | 6 438 | **WSL2 TF-GPU diagnostic** — the documented remedy for the `LD_LIBRARY_PATH` failure |
| `src/sep_cam_probe.py` | 18 365 | **CAM validation probe** — the audio-loading semantics the engine mirrors (`multisensory_engine.py:228-229`) |
| `src/shift_example.py` | 1 460 | upstream example |
| `src/videocls.py` | 16 322 | action-recognition path, unused |

`multisensory/src/__init__.py` (150 bytes) is also never imported — the engine
puts `src/` *on* `sys.path` and imports flat names, so `src` is not a package.

**Recommendation to raise:** vendor `cam_analyze.py`, `sep_cam_probe.py` and
`gpu_doctor.py` anyway (46 105 bytes). They are not runtime dependencies, but
they are the artefacts the engine's correctness comments cite as the
comparability baseline, and the diagnostic the environment documentation depends
on. Dropping them severs the evidence chain for a saving of 45 KB. Decision
required.

### 2.5 Static-vs-instrumented diff

| | static | instrumented |
|---|---|---|
| modules found | 13 | 14 |
| difference | — | `aolib` (the package `__init__`) |

The single discrepancy is a limitation of the static tool (it resolves
`import aolib.util` to the submodule and does not add the implicit parent package
import), not a dynamic import. **No module was reached dynamically that static
analysis missed.** There is no hidden `importlib`, `__import__`, or plugin
lookup in this dependency set.

Lazy imports that exist in the reached files and are **never executed**, listed
because they would be a vendoring trap if any of them were ever called:

| in | lazily imports | present in repo? |
|---|---|---|
| `sourcesep.py` | `sep_eval`, `sep_i3d` | **no — these modules do not exist** |
| `shift_net.py` | `multi_pass_optimizer` | **no — does not exist** |
| `aolib/sound.py` | `pylab`, `subband` | `subband` yes, `pylab` is matplotlib |
| `aolib/img.py` | `matplotlib` | third-party |
| `aolib/util.py` | `billiard`, `pathos`, `sklearn`, `statsmodels`, `networkx`, `msgpack`, `msgpack_numpy`, `h5py`, `cv`, `parula`, `iputil`, `pstats`, `pylab` | mostly not installed |

Three of these targets are already missing from the upstream repository today.
That is pre-existing, and under the "no behaviour changes" rule it stays that
way; it is recorded here so the vendoring is not blamed for it later.

### 2.6 Runtime data dependencies outside the `.py` files

Searched the reached set for file reads. Hits are all Andrew Owens' original
training-machine paths — `shift_params.py:33,36-37,107,115,120-121,203,205-206`
(`/data/scratch/owens/...`), `sep_dset.py:7` and `shift_dset.py:9`
(`cifar_path = "../data/cifar-10"`), `sep_params.py:22` (`"/results/nets/sep"`).
None is opened on the inference path — the instrumented run completed with none
of those paths existing. The engine overrides `pr.model_path` explicitly
(`multisensory_engine.py:194`).

**Conclusion: the vendoring set is the 14 `.py` files plus the checkpoint. Nothing else.**

---

## 3. Provenance and licence (2c)

### 3.1 multisensory — Apache License 2.0

`multisensory/LICENSE` is present, 11 365 bytes, full Apache 2.0 text, with the
appendix filled in:

```
Copyright [2018] [Andrew Owens and Alexei A. Efros]
```

Vendoring is permitted. Apache 2.0 §4 obliges us to: ship a copy of the licence
(4a), mark modified files (4b), retain existing copyright/attribution notices
(4c), and carry the NOTICE file if one exists (4d). There is **no** `NOTICE`
file in the repository — checked; only `LICENSE`. `NOTICES.md` at the backend
root will therefore be our own attribution file, not a reproduction of an
upstream NOTICE.

### 3.2 multisensory — divergence from upstream

The fork's history is 28 commits. The last upstream commit is:

```
8a5b44a  2018-11-08  Andrew Owens  fix windows compatibility bug
```

`ca7177c` (2018-04-14) … `8a5b44a` (2018-11-08) — 16 commits — are
`andrewowens/multisensory`. **13 commits diverge**, all 2026, by `mysterI0s` and
`theHATEM`:

| commit | date | subject |
|---|---|---|
| `120ab3e` | 2026-03-02 | migrated the repository to python 3.13 |
| `602555a` | 2026-03-04 | Refactored Code |
| `e65fefe` | 2026-03-04 | fixed tf.contrib |
| `994bea0` | 2026-03-04 | Made the model work |
| `ae452d6` | 2026-03-06 | Added the full implementation plan to migrate from raw tensorflow to Pytorch |
| `450a24b` | 2026-03-08 | Converted the repo from tf to pytorch |
| `fee05fd` | 2026-03-08 | Fixed the convert weights file |
| `1c97ef4` | 2026-03-08 | Fixed conversion skips |
| `ec35da6` | 2026-07-30 | Fixed some bugs |
| `3d8e276` | 2026-07-30 | Fix bugs |
| `1fa9728` | 2026-08-03 | Tested the model on GTX 1660 TI |
| `31c0cd2` | 2026-08-04 | Repo cleanup for server integration (phase 3) |
| `e94324a` | 2026-08-04 | Docs: SonicSight server integration section (phase 6) |
| `a8b1f37` | 2026-08-04 | Make aolib's matplotlib imports lazy — server loads without a plot stack |
| `eb75ea5` | 2026-08-13 | removed all the migration scripts used to migrate the repository from 2.7 to 3.13 and from tf to pytorch |

Summary of the divergence: **Python 2.7 → 3.13 port; `tf.contrib` removal;
migration to `tensorflow.compat.v1` with `tf.disable_v2_behavior()`; a PyTorch
port whose scripts were subsequently deleted; server-integration changes
(GPU session config from `MS_GPU_*`, the `TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512`
default, lazy matplotlib).** The code we vendor is therefore *not* upstream
2018 code — it is upstream plus a substantial port. Apache 2.0 §4(b) requires
these modified files to carry prominent notices stating that they were changed;
they currently do not. That obligation lands on us at vendoring time and is
satisfied by `PROVENANCE.md` + a marker, not by editing 14 files.

`31c0cd2` also **deleted upstream's `README.md`** from this fork
(`/home/hatem/multisensory/README.md` — an older clone — still has it). The
paper citation and the model-download instructions therefore no longer travel
with the code. `REPORT.md` and `ORIGINAL_REPORT_for_reference.md` remain.

Upstream paper: Owens & Efros, *Audio-Visual Scene Analysis with Self-Supervised
Multisensory Features*, ECCV 2018 (arXiv:1804.03641v2) — cited in
`src/engines/multisensory_engine.py:1`.

### 3.3 Sound of Pixels — **no licence, no provenance, already vendored**

This is the finding that needs a decision before anything else.

The Sound of Pixels model code is **already inside the backend repository**,
copied without attribution:

| path | bytes | content |
|---|---|---|
| `src/models/__init__.py` | 3 280 | `ModelBuilder` — verbatim upstream class |
| `src/models/audio_net.py` | 3 976 | `Unet` |
| `src/models/vision_net.py` | 4 032 | `ResnetFC`, `ResnetDilated` |
| `src/models/synthesizer_net.py` | 2 342 | `InnerProd`, `Bias` |
| `src/models/criterion.py` | 1 285 | `BCELoss`, `L1Loss`, `L2Loss` |
| `src/utils/*.py` | 11 924 | STFT helpers, video transforms |

Searched `src/models/`, `src/utils/` and `src/inference.py` for
`copyright`, `licen[sc]e`, `MIT`, `Sound of Pixels`, `hangzhao`:

- **No `LICENSE` file anywhere in `SonicSightBackend`.**
- **No copyright header in any of those files.**
- **No upstream URL and no upstream commit SHA recorded anywhere.**
- The only trace is prose: `config.py:61` "(Sound of Pixels paper, Section 3.3…)",
  `model_registry.py:96`, `inference.py:79,226,824,1257`.
- `README.md:185` says, verbatim: *"Distribute your license here. All rights
  reserved by the original project contributors."*

So the state today is: third-party research code redistributed in a **public**
repository with no licence of its own, no attribution, and no provenance record.
The upstream project is `hangzhaomit/Sound-of-Pixels` (MIT CSAIL). Its licence
was **not verified** in this pass — the repository is not checked out locally and
no network fetch was made. **Not measured.**

**This must be resolved before Phase 1.** Required: (i) fetch the upstream
`Sound-of-Pixels` repository, record its licence and the commit the code
corresponds to; (ii) add the licence text and attribution; (iii) if the upstream
licence is MIT (widely reported, unverified here), MIT §1 requires the copyright
notice and permission notice in all copies — which the repository currently does
not have. The `README.md:185` sentence is not a licence and should be replaced.

Note this is a **pre-existing defect of the backend repository**, discovered
during Phase 0. Under the "record, do not fix" rule it belongs in the defect
ledger as a new entry rather than being silently patched — but unlike a code
bug, leaving it unfixed while the repository stays public is itself the problem.
Decision required.

---

## 4. Checkpoints and weights (2d)

### 4.1 Required by the backend today

| file | bytes | sha256 | lives in | origin |
|---|---|---|---|---|
| `src/ckpt/sound_best.pth` | 121 145 339 | `c1d45b3a9cb469dbbb139addae56ed6581b206e9d0018e0d8a34908660298f75` | backend (untracked) | Sound of Pixels release — **exact source not recorded in-repo** |
| `src/ckpt/frame_best.pth` | 45 356 134 | `33b22fb447eaa10ec2c0bef98949e042cf75c2f6e28549803b6782eab1d18fa0` | backend (untracked) | as above |
| `src/ckpt/synthesizer_best.pth` | 652 | `fa29d150d72a8c7b5e0d612d52629d4642d2bd13fa43dab87261da0f906e806e` | backend (untracked) | as above |
| `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | 46 830 571 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | **user home, outside every repo** | torchvision `ResNet18_Weights.IMAGENET1K_V1`, downloaded on first engine load (§1.4) |
| `multisensory/results/nets/sep/full/net.tf-160000.data-00000-of-00001` | 1 364 209 708 | `baaea29b6d1b887258ab3cded4972e74dcdc3f99138f751de0d3ac2693c2996c` | **sibling repo** | `people.eecs.berkeley.edu/~owens/multisensory-nets.zip` (`download_models.sh`) |
| `multisensory/results/nets/sep/full/net.tf-160000.index` | 25 545 | `e895db694d6b92ae0da3fded37b914d8da14169a4386b9c585a7a8a79fe7ff90` | sibling repo | as above |
| `multisensory/results/nets/sep/full/net.tf-160000.meta` | 13 016 751 | `ac60a670c2075b8bc4b57aedd0d3f8c21e98014e97dbcc4938307a31e603dad5` | sibling repo | as above |

**Total actually required: 1 590 584 700 bytes ≈ 1.48 GiB** (1.28 GiB of it the
multisensory separation checkpoint).

### 4.2 Present but not required

`multisensory/results/` holds **4 181 037 818 bytes (3.89 GiB)** in total. The
other four checkpoints — `nets/cam/net.tf-675000`, `nets/sep/large/net.tf-900000`,
`nets/sep/unet-pit/net.tf-160000`, `nets/shift/net.tf-650000` — are **2.60 GiB
that the backend never loads**. `MULTISENSORY_CHECKPOINT` could point at them, but
`MULTISENSORY_SPEC` and the engine's `pr.cam is False` guard
(`multisensory_engine.py:189-192`) mean only `sep/full` is a supported target.

The checkpoint provisioning script (Phase 1 §3c) should fetch **only `sep/full`**
if the upstream zip permits partial extraction, or extract-then-prune if it does
not. Whether `multisensory-nets.zip` still resolves at that Berkeley URL was
**not tested** (no network fetch was made in this pass).

### 4.3 Docker consequence

1.48 GiB of weights must **not** be baked into the image (brief §3c agrees).
Volume-mount `./checkpoints` read-only. `resnet18-f37072fd.pth` must be
provisioned into the same mechanism via `TORCH_HOME`, or the `sonicsight` engine
cannot load offline (§1.4).

---

## 5. Runtime stack, exactly (2e)

Measured by importing each framework inside the project venv and reading its own
build metadata.

| item | value | how established |
|---|---|---|
| host OS | Windows 10 Pro 19045 + WSL2 Ubuntu, kernel `6.18.33.2-microsoft-standard-WSL2` | `uname -a` |
| Python | **3.12.13** (uv-managed CPython, `/home/hatem/.venv`) | `sys.version` |
| environment count | **one** venv, both frameworks, one interpreter | `pip freeze` |
| TensorFlow | **2.21.0** | `tf.__version__` |
| TF API style | `tensorflow.compat.v1` + `tf.disable_v2_behavior()` — **TF 2.x, not 1.x** | `sep_video.py:5,7` |
| `tf.contrib` | **absent and not used** (`hasattr(tf,"contrib") == False`) | probe |
| TF CUDA build | `cuda_version 12.5.1`, `cudnn_version 9`, `is_cuda_build True` | `tensorflow.python.platform.build_info` |
| TF compute capabilities (declared) | `sm_60, sm_70, sm_80, sm_89, compute_90` | same |
| PyTorch | **2.11.0+cu128**, CUDA 12.8, cuDNN 9.19.0 (`91900`) | `torch.__version__` etc. |
| torchvision | 0.26.0+cu128 | `pip freeze` |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 144 MiB, **compute capability 7.5** | `nvidia-smi`, `torch.cuda.get_device_capability` |
| driver | NVIDIA-SMI **610.57.01**, KMD 610.88, CUDA UMD **13.3** | `nvidia-smi` |
| TF GPU visibility | **works, but only with the venv `LD_LIBRARY_PATH` fix** | below |
| server location | **inside WSL2**, not on the Windows host | see §5.2 |
| supporting libs | `tf_keras 2.21.0`, `tf-slim 1.1.0`, `keras 3.15.1`, `numpy 2.3.5`, `scipy 1.18.0`, `opencv-python 5.0.0.93`, `grpcio 1.83.0`, `protobuf 7.35.1`, `librosa 0.11.0`, `moviepy 2.2.1`, `fastapi 0.141.1`, `uvicorn 0.52.1` | `pip freeze` |

### 5.1 The `LD_LIBRARY_PATH` condition

Bare `python -c "import tensorflow"` reports **zero GPUs**:

```
Cannot dlopen some GPU libraries ... Skipping registering GPU devices...
gpus []
```

The venv's `activate` script appends:

```sh
NV_LIBS=$(find "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia -maxdepth 2 -name lib -type d | sort | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

With it, `gpu_doctor.py --tf` reports all eleven CUDA libraries dlopen-able and:

```
GPUs : [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**This is an environment activation dependency, not a code dependency.** It must
be reproduced explicitly in any container `ENV LD_LIBRARY_PATH=...`, because a
container does not source `activate`.

### 5.2 The WSL2 boundary

The whole server process runs inside WSL2 Ubuntu. The Windows host contributes
only the GPU driver, projected into WSL at `/usr/lib/wsl/lib/libcuda.so.1`.
`README.md:169-170` records that phones cannot reach a WSL2 server without
`networkingMode=mirrored` or a `netsh portproxy` rule for 50051 — the boundary is
crossed at the network layer, on the client side.

### 5.3 Discrepancy worth stating

TF 2.21's declared compute capabilities do **not** include `sm_75`, yet the
instrumented inference in §2.1 ran on the sm_75 card and produced output
(`Created device ... compute capability: 7.5`, `Loaded cuDNN version 91900`,
`INFERENCE OK`). The empirical result is what it is; the declared list and the
observed behaviour disagree, and this document does not attempt to explain why.
No timing comparison was made — **not measured**.

---

## 6. Docker feasibility (2f)

### 6.1 The question the brief asks

> Can one image carry both frameworks with GPU support on a single CUDA version —
> or must GPU stay on the host?

**One image can.** The disqualifier the brief anticipated does not exist here.

The concern was TF 1.x pinning to CUDA 10.x. This project does **not** run TF 1.x.
It runs **TensorFlow 2.21.0** using the `tf.compat.v1` API with
`disable_v2_behavior()`. `tf.contrib` is gone (`e65fefe fixed tf.contrib`) and
absent from the installed TF. **No NGC `tensorflow:*-tf1-py3` image is needed, and
CUDA 10.x is not in play.**

Further, neither framework needs a CUDA *base image* at all. Both ship their CUDA
userspace as pip wheels (`nvidia-*-cu12`, fifteen wheel lib directories confirmed
by `gpu_doctor.py`), and both bind to the driver's `libcuda.so.1`, which under
WSL2 comes from the host projection. TF's wheel is built against CUDA 12.5.1 and
torch's against 12.8; they already coexist in **one venv, one interpreter, one
process** on this machine — which is the arrangement ADR-0004 mandates and
measured against. A container that reproduces that venv reproduces the same
arrangement.

### 6.2 Options evaluated

**A — CPU-only image.** Both frameworks, no CUDA wheels. Builds and runs
anywhere, including a CI runner with no GPU. Inference is slow; GPU runs stay on
the host. Small enough to build repeatedly. *Cost: none to correctness; the
container cannot produce comparable performance numbers, which the brief already
forbids anyway.*

**B — Single GPU image.** Technically **feasible** (§6.1). One image, both
frameworks, both GPU-capable, honouring ADR-0004's single process. *Costs:* the
image carries the full CUDA wheel set for both frameworks — **not measured**, but
clearly multi-gigabyte; requires NVIDIA Container Toolkit under WSL2; requires
the `LD_LIBRARY_PATH` line reproduced as `ENV`; and CI cannot smoke-test it on a
GPU-less runner. Also inherits the sm_75 discrepancy of §5.3.

**C — Two engine images plus a gateway.** **Recommend rejecting on documented
grounds, not on cost.** `ADR-0004 — One process hosting both model frameworks
rather than separate services` is an *accepted* ADR that explicitly records this
design as *"the abandoned sidecar design (separate CAM and separation services on
ports 5556/5557), which was deleted before implementation"*. Its stated reason:
each framework's CUDA context and cuDNN kernel images cost ~1–1.3 GB per
framework per process, *"On the 6 GB measurement-environment card, that
difference is the difference between both models fitting and not fitting."*
Splitting into two containers pays that overhead twice and reverses a recorded
architectural decision that the measurement record depends on (NFR-PERF-010,
FR-023, FR-064). This is not a scope question — it is a decision the project has
already made and documented.

### 6.3 Recommendation

**A for CI; B documented and buildable; GPU execution stays on the host.**

- CI builds and smoke-tests the **CPU-only** image on the self-hosted runner. It
  must pass with no GPU present.
- A GPU variant (option B) is provided behind a build argument and documented in
  `docs/DOCKER.md`, including the `deploy.resources.reservations.devices` stanza,
  the NVIDIA Container Toolkit requirement under WSL2, and the CPU fallback when
  the toolkit is absent.
- All GPU measurement stays where the measurement record was made: the host,
  manually. Stated as a limitation, not papered over.

Prerequisite, currently unmet: **Docker is not reachable.**

```
$ wsl docker version
The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.

PS> docker version
error during connect: ... open //./pipe/dockerDesktopLinuxEngine:
The system cannot find the file specified.
```

`wsl -l -v` shows `docker-desktop  Stopped`. Docker Desktop must be started and
WSL integration enabled for the Ubuntu distro before Phase 2 can begin.

### 6.4 Build context

| directory | size |
|---|---|
| `SonicSightBackend` total | 192 M |
| `src/ckpt` | 159 M |
| `loadtest/results` | 26 M |
| `.git` | 5.5 M |
| **remainder (the intended context)** | **≈ 1.5 M** |

With `.dockerignore` excluding those three, the context is ~1.5 MB plus the
~300 KB of vendored multisensory source. Comfortably under the brief's threshold.

---

## 7. Baseline measurements taken in this pass

For the Phase 1 regression oracle. Both were run; both are reproducible by the
command shown.

**Unit suite** — `python -m pytest tests -q`, in the project venv, from
`SonicSightBackend/`:

```
79 passed, 1 skipped, 5 warnings in 10.43s
```

80 tests collected. The single skip is
`tests/test_multisensory_engine.py:138` — *"checkpoint present; this test is for
the empty environment"* — i.e. it is the FR-024 empty-environment test, skipped
precisely because this host has the checkpoint.

**Replay harness** — **not run.** See §8.2 for why it cannot be run yet.

---

## 8. Where the brief and the repository disagree

Seven items. Each needs a decision before Phase 1.

### 8.1 "79 unit tests"

The suite **collects 80** and reports **79 passed, 1 skipped**. The brief's
figure is the pass count, not the test count. Cosmetic, but the phrasing
"79 unit tests" will not survive an examiner running `--collect-only`.

### 8.2 "Run the replay harness on both models" — the harness cannot do this

Two independent blockers.

1. **No model selection.** `replay_client.py` has exactly six arguments —
   `--video`, `--host`, `--out`, `--speed`, `--max-seconds`, `--no-analyze`
   (`replay_client.py:334-340`). It sends **no gRPC metadata at all** (searched;
   no `metadata` and no `sonicsight-model` anywhere in the file), so every replay
   run takes the server default, `sonicsight`. The project's own documentation
   agrees: UC-18 is recorded as *"Implemented and validated … **Halves mode
   only**"*, and `TESTPLAN.md:94` states that deterministic multisensory replay
   *"needs a `--model multisensory` mode"* which does not exist.

2. **No input clip.** `replay_client.py` requires `--video`. There is no video
   file in this workspace: `SonicSightBackend/data/` does not exist in the
   `final_project` copy, and in the older `/home/hatem/SonicSight` copy it exists
   and is **empty**. `loadtest/run_func001.sh:45` references
   `../data/test_instruments.mp4`, which is not present.

Consequence for the brief's §3a gate: the byte-identical replay oracle can be
established for `sonicsight` (halves) **only**, and only once a reference clip is
supplied. For `multisensory` there is no replay path to be byte-identical
against. Options: (a) supply a clip and gate on halves only, stating the speech
path is covered by the unit suite plus a manual check; (b) add `--model` to
`replay_client.py` — which is a behaviour change to the harness the brief forbids
mid-vendoring, and would need its own before/after baseline; (c) build a separate
engine-level determinism check for the speech path (load, one fixed window,
hash the output arrays) that does not touch the harness. **(c) is the cheapest
honest option and does not disturb UC-18.** Decision required.

### 8.3 The workspace is a copy, and the documentation is not in it

`\\wsl.localhost\Ubuntu\home\hatem\final_project\SonicSight` was created
**2026-08-13 15:33** — today. The original workspace is `/home/hatem/SonicSight`,
and it contains a fourth item the copy does not: **`docs/`, a separate git
repository** holding `ANALYSIS_REPORT.md` (166 KB), `TECHNICAL_REPORT.md`
(210 KB), `nfr/nfr_targets.yaml`, `adr/`, `diagrams/`. Every requirement id the
brief cites — FR-024, FR-066, UC-13, UC-18, FR-P01…P08 — is defined there and
**nowhere in the `final_project` copy**. `TECHNICAL_REPORT.md` contains none of
them; `ANALYSIS_REPORT.md` contains 36.

Also absent from the copy: `data/` and `replay_out/`.

Consequences: (i) the load suite's E8/E10/E12/E14 escapes resolve to nothing in
this workspace; (ii) `docs/VENDORING_INVENTORY.md` as the brief names it is
ambiguous — it could mean the sibling `docs` repository. **This file has been
written to `SonicSightBackend/docs/VENDORING_INVENTORY.md`**, inside the backend,
which is what "self-contained" implies. Confirm that is intended.

### 8.4 Self-containment is not one dependency, it is three

The brief frames the task as breaking the coupling to `multisensory`. The backend
also reaches into the **mobile** repository (E9, E11, E13) and the **docs**
repository (E8, E10, E12, E14). Severing those is not vendoring — E13 *is*
NFR-COMPAT-001 (the byte-identical proto check) and E8/E10 supply the NFR
thresholds the load suite asserts against. Cutting them changes what the
measurement harness measures.

Recommended scope: **make the server self-contained (E1–E6); leave the load suite's
sibling reads intact but degrade gracefully when the siblings are absent** —
`scenarios.py:820` already guards `MOBILE_PROTO` with `os.path.exists`; `nfr.py:25`
and `runmeta.py:62` do not. A fresh clone then serves both models with no siblings
present (Goal 1 satisfied) while the campaign harness keeps measuring exactly what
it measured. Decision required.

### 8.5 The Sound of Pixels licence gap

§3.3. Third-party research code in a public repository with no licence, no
attribution, no upstream SHA. Pre-existing; needs a decision, not a silent patch.

### 8.6 `.gitignore` will fight the deliverables

`SonicSightBackend/.gitignore` contains a bare `*.txt`. `requirements.txt` is
already tracked so it is unaffected, but **`checkpoints/MANIFEST.sha256` is fine
while `requirements-dev.txt` is not** — it will need `git add -f` or a negation
rule. Flagging so it is not discovered as a mystery in Phase 1.

### 8.7 The tag the brief asks for, on which HEAD

Phase 1's first action is to tag `main` HEAD as `measured/backend-em`. That SHA is
**`17016dd350ec93f6c5941a325632fceb2c8ee0ab`**, and `origin/main` points at the
same commit, so the tag is unambiguous. But the measurement campaign's own
artefacts live in `loadtest/results/em*` in this repository, and the older
workspace's backend sits on branch `em-verification` at a *different* commit
(`2b6485e`). If any campaign number was produced from that working tree rather
than from `17016dd`, the tag documents the wrong commit. Worth one confirmation
before tagging, since the tag is the anchor for every number in the report.

---

## 9. Time estimates

Estimates, not measurements. They assume the decisions in §8 are made first and
that Docker Desktop is running.

| phase | estimate | dominated by |
|---|---|---|
| 0 — inventory (this document) | **complete** | — |
| 1 — vendoring | 3–5 h | the regression oracle, not the file copy. Copying 14 files and rewriting E1–E6 is under an hour; establishing a defensible before/after gate (§8.2) and re-running the fresh-clone acceptance test is the rest |
| 2 — Docker | 4–7 h | first `pip install` of TF + torch into an image, iterated. Cold build time is the long pole and is **not estimated** — it will be measured |
| 3 — CI/CD | 3–4 h | runner registration and the first genuinely green end-to-end run; the workflow itself is short |
| 4 — documentation | 2–3 h | `report_insert.md` and `DOCKER.md` carry real numbers, so they trail Phase 2's measurements |
| **total** | **12–19 h** | excludes any decision latency in §8 |

Phase 1 slips if §8.2 lands on option (b). Phase 2 slips if the GPU variant is
required for CI rather than documented.

---

## 10. Commands used

Reproducible; each produced output quoted above.

```powershell
# repository state
wsl -e bash -c 'cd .../SonicSightBackend; git rev-parse HEAD; git rev-parse origin/main; git status --porcelain'
wsl -e bash -c 'cd .../multisensory; git log --format="%h|%ad|%an|%s" --date=short'

# static import graph          (script: /tmp/p0_static.py)
wsl -e /home/hatem/.venv/bin/python /tmp/p0_static.py

# instrumented real inference  (script: /tmp/p0_trace.py)
wsl -e bash -c 'source /home/hatem/.venv/bin/activate; python /tmp/p0_trace.py 0'

# environment                  (script: /tmp/p0_env.py)
wsl -e /home/hatem/.venv/bin/python /tmp/p0_env.py
wsl -e bash -c 'source /home/hatem/.venv/bin/activate; cd .../multisensory/src; python gpu_doctor.py --tf'

# checkpoints
wsl -e sha256sum <the seven files in §4.1>

# baseline suite
wsl -e bash -c 'source /home/hatem/.venv/bin/activate; cd .../SonicSightBackend; python -m pytest tests -q'
```
