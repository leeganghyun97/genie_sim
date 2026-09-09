# G2 recurrent visual pre-loss v2 reproduction

This package reproduces the strongest historical strict-lift configuration
without loading its old actor, critics, optimizer, or replay. The historical
120k checkpoint is an evaluation reference only:

```text
SHA-256 0aa8f2d381b527617dcae51169319b923e9ad1a76538e2c7b31e8820cb9dc18e
```

The fresh recipe is sealed by
`configs/model/g2_recurrent_visual_preloss_v2.json`:

- GRU hidden/layers: `256/1`
- sequence/burn-in/stride: `16/4/12`
- gradient clip/batch: `5.0/64`
- temporally consistent RGB-D shift: `2 px`
- cross-camera contrastive loss: `0`
- temporal pose-residual loss: `0`

The older relative-pose, shifted-view pose-consistency, contact,
depth-validity, and student-head objectives remain enabled at their recorded
baseline weights. "Pre-loss" only disables the two later experimental losses.

## Clone and verify

Publish the prepared branch to a user-owned fork. The helper refuses the
upstream `origin` URL and includes the referenced Git LFS objects through the
pre-push hook:

```bash
bash scripts/publish_g2_preloss_v2_branch.sh \
  --fork-url git@github.com:YOUR_ACCOUNT/genie_sim.git
```

Clone the reproducibility branch and fetch the binary USD layers:

```bash
git clone --branch codex/g2-preloss-v2-repro <repository-url> genie_sim
cd genie_sim
git lfs pull
python scripts/check_g2_preloss_v2_clone_readiness.py --require-tracked
```

The checker verifies the G2 USD layers and canonical URDF by SHA-256, resolves
all referenced USD layers, parses the URDF, compiles the required Python
source, checks the sealed recipe, and requires every dependency to be in the
Git index.

Isaac Sim, Isaac Lab, demonstrations, checkpoints, replay, and outputs are not
committed. Their contract is recorded in the two JSON files under
`configs/repro`. Keep the five demonstration HDF5 files in the order-independent
colon-separated variable below; the learner records and checks their hashes.

## Fresh training

```bash
export G2_VISUAL_TEACHER_PYTHON=/absolute/path/to/isaac/python
export G2_VISUAL_TEACHER_OUTPUT_ROOT=/absolute/path/to/output
export G2_DEMONSTRATION_DATASETS=/absolute/a.hdf5:/absolute/b.hdf5:/absolute/c.hdf5:/absolute/d.hdf5:/absolute/e.hdf5

bash scripts/run_g2_recurrent_visual_teacher_preloss_v2.sh \
  --wandb --wandb-mode online \
  --wandb-project geniesim-g2-recurrent-visual-teacher \
  --wandb-run-name g2-preloss-v2-fresh
```

The wrapper rejects `--resume`, `--checkpoint`, and `--append`. It also rejects
overrides to the model, augmentation, and auxiliary-loss settings that define
this comparison. Runtime scale options such as `--num-envs` and
`--total-transitions` can still be lowered for a smoke test.

To inspect the fully resolved child command without starting Isaac Sim:

```bash
G2_PRELOSS_DRY_RUN=1 \
  bash scripts/run_g2_recurrent_visual_teacher_preloss_v2.sh --no-wandb
```
