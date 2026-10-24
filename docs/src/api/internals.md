# Internals

```@meta
CurrentModule = RANSACScoringDoneRight
```

Declared `public` because downstream packages (the paper scripts, the
evaluation packages) reach into them, but low-level: the sweep itself,
workspace builders and the accessors of its buffers. They are not a stable
interface.

## The sweep and its workspace

```@docs
sweep!
default_scoring_workspace
alloc_scoring_workspace
_build_lg_table
```

## Accessors

```@docs
sweep_order
penalties
sort_key
lg_table
```

## Helpers

```@docs
_predictive_score_penalty
```

## Not public

Documented internals that the public docstrings refer to. They are not part of
the API.

```@docs
RANSACScoringDoneRight.RadixPermBuffers
RANSACScoringDoneRight._radix_sortperm!
RANSACScoringDoneRight._radix_key
RANSACScoringDoneRight._build_prior_table
Base.resize!(::ScoringWorkspace, ::Integer)
RANSACScoringDoneRight.PrecomputedPair
RANSACScoringDoneRight.ig_sweep_kernel!
RANSACScoringDoneRight.ScaleMarginal
RANSACScoringDoneRight.SmoothTLSMarginal
RANSACScoringDoneRight._PredictiveStudentTUniform
RANSACScoringDoneRight._build_score_weight_matrix
```
