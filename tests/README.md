# Tests

```bash
pip install pytest
pytest tests -m "not golden"    # synthetic tests, no data needed
```

## Synthetic tests

`test_repair_synthetic.py` builds small phantoms in memory: a straight tube cut by a
gap, plus a triangulated tube standing in for a fitted mesh. They pin the properties
the method claims, namely that repair reconnects components through the mesh, only
ever adds voxels, declines bridges that exceed the foreground-growth budget, and
leaves an already-connected mask untouched. They need no dataset and take a few
seconds.

## Golden regression tests

`test_golden_repair.py` replays finished runs and compares the repaired masks with
the stored ones, voxel for voxel. Because every run records its own repair
parameters in `metrics.json`, a run directory is self-describing: the test rebuilds
the command line from it, re-runs `repair_case` into a temporary directory, and
diffs the result. Repair loads a fitted mesh rather than fitting one, so it is
deterministic and the comparison is exact.

These fixtures are nnU-Net predictions derived from the SEGA, AortaSeg24, TopCoW and
PARSE datasets, whose terms do not allow redistribution, so they are not part of this
repository. To run them against your own results:

```bash
cp tests/golden_manifest.example.json tests/golden_manifest.json
# edit the "runs" list to point at your run directories
pytest tests -m golden
```

Without `golden_manifest.json` these tests skip, so the suite stays green on a clean
checkout. `golden_manifest.json` is git-ignored.
