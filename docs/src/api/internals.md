# Internals

```@meta
CurrentModule = RANSACScoringDoneRight
```

Declared `public` because downstream packages (the paper scripts, the
evaluation packages) reach into them, but low-level. They are not a stable
interface.

The sweep, its workspace builders and accessors moved to
VisualGeometryToolkitCore with `StudentTUniformScoring` (2026-11-11).

## Not public

Documented internals that the public docstrings refer to. They are not part of
the API.

```@docs
RANSACScoringDoneRight.PrecomputedPair
RANSACScoringDoneRight.ScaleMarginal
RANSACScoringDoneRight.SmoothTLSMarginal
RANSACScoringDoneRight._build_score_weight_matrix
```
