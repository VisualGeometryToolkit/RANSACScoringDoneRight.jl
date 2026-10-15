# Internals

```@meta
CurrentModule = RansacScoringDoneRight
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
RansacScoringDoneRight.RadixPermBuffers
RansacScoringDoneRight._radix_sortperm!
RansacScoringDoneRight._radix_key
RansacScoringDoneRight._build_prior_table
Base.resize!(::ScoringWorkspace, ::Integer)
RansacScoringDoneRight.PrecomputedPair
RansacScoringDoneRight.ig_sweep_kernel!
RansacScoringDoneRight.ScaleMarginal
RansacScoringDoneRight.SmoothTLSMarginal
```
