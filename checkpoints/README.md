# Released Stage-III models

`stage3/` contains the final 30-program population for every reported
Stage-III run: six semantic pairs, two methods, and five paired training seeds
(60 runs and 1,800 NCA models in total).

The directory layout is:

```text
stage3/<semantic-pair>/<strategy>/seed_<seed>/
|-- run_manifest.json
|-- summary.json
|-- sub_00_w[...]/final.pt
|-- ...
`-- sub_29_w[...]/final.pt
```

`run_manifest.json` records the training configuration. `summary.json` is a
compact release record containing the 30 preference weights and the canonical
objective vectors used by `results/stage3`. It excludes local paths, optimizer
state, intermediate checkpoints, and training diagnostics.

The Cactus--Maple summaries contain the final prompt-normalized reevaluation
with `a tall green cactus` and `a red maple tree`. This distinction matters
because earlier local visualization caches used a different prompt assignment;
those caches are not part of the released evidence.

`stage3/manifest.csv` lists all runs and their relative directories.
`stage3/SHA256SUMS` covers every released model and metadata file. Verify the
release from the repository root with:

```bash
python scripts/verify_stage3_checkpoints.py
```

The `final.pt` files contain model weights only. Loading and rollout require
the pinned PyTorch environment described in the main README.
