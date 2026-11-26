# Internals

```@meta
CurrentModule = RANSACScoringDoneRight
```

Declared `public` because downstream packages (the paper scripts, the
evaluation packages) reach into them, but low-level. They are not a stable
interface.

The sweep, its workspace builders and accessors moved to
VisualGeometryToolkitCore with `StudentTUniformScoring` (2026-11-11).

## The configuration converter

`configs/default.toml` holds the defaults of record, read through the package's
versioned converter (VisualGeometryToolkitCore's `ConfigVersions`).

```@docs
RSDR_CONFIG_VERSIONS
```

## Not public

Documented internals that the public docstrings refer to. They are not part of
the API.

```@docs
RANSACScoringDoneRight.PrecomputedPair
RANSACScoringDoneRight._build_score_weight_matrix
```
