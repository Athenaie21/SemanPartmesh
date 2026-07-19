# SemanPartmesh

Semantic-guided quadrilateral meshing pipeline integrating **PartField** (semantic features), **NeurCross** (cross field training), and **quad_extract** (MIQ + libQEx extraction).

## Directory Layout

```
SemanPartmesh/
├── run_pipeline.py          # Main pipeline entry (PartField → NeurCross → extract)
├── extract_quad.py          # Stage 3: quad mesh extraction wrapper
├── build_complexity_map.py  # Semantic complexity / size field utilities
├── visualize_feature_clusters.py
├── requirements.txt
│
├── scripts/                 # Shell launchers and environment setup
│   ├── install_env.sh
│   ├── run_pipeline_simplified.sh
│   ├── run_baseline*.sh
│   └── ...
│
├── eval/                    # Evaluation metrics (angle distortion, Jacobian, etc.)
├── PartField/               # PartField source (vendored)
├── NeurCross/               # NeurCross source (vendored)
├── quad_extract/            # C++ quad extraction (CMake)
│
├── data/
│   └── scratch/             # Temporary / test meshes and crossfields
│
├── experiments/             # Historical experiment outputs
│   ├── compare/             # Single-mesh baseline vs pipeline comparisons
│   ├── reconstruction/      # Reconstruction dataset batch runs
│   ├── hunyuan/             # Hunyuan patch extraction script + results
│   │   └── run_hunyuan_patch_extract.py
│   ├── pipeline/            # Standard pipeline outputs
│   ├── baseline/            # Baseline-only outputs (created at runtime)
│   └── cheburashka/         # Complexity analysis outputs
│
└── docs/
    └── exp.md               # Experiment notes
```

## Quick Start

```bash
# Install environment
bash scripts/install_env.sh

# Run simplified pipeline on a mesh or directory
bash scripts/run_pipeline_simplified.sh input/armadillo.obj

# Run baseline (NeurCross only, no PartField guidance)
bash scripts/run_baseline_simplified.sh input/
```

## Pipeline Stages

1. **PartField** — extract semantic features (`partfield` conda env)
2. **NeurCross** — train cross field with optional semantic guidance (`neurcross` conda env)
3. **extract_quad.py** — MIQ parametrization + libQEx quad mesh generation

## External Dependencies (not in repo)

These must be set up separately before running the full pipeline:

| Path | Purpose |
|------|---------|
| `PartField/model/model_objaverse.ckpt` | PartField checkpoint |
| `libigl/`, `libQEx/` | C++ dependencies for `quad_extract` |
| `Baseline/NeurCross/` | Baseline training scripts |
| `instruction_guidance/` | Instruction-guided mode dataset + metadata |
| `input/` | Default input mesh directory |

## Output Location

New pipeline runs write to `experiments/pipeline/pipeline_output/` by default (with optional timestamp suffix). Override with `--output_dir`.

See also: [脚本说明（中文）](docs/脚本说明.md) — detailed guide for all scripts and workflows.
