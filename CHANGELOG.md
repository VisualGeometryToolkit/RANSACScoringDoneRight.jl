# RANSACScoringDoneRight changelog

## 0.4.1 (unreleased)

VisualGeometryToolkitCore 0.6 (compat `"0.6"`): the breaking upstream release; no
change to this package's own code or public API. None of the names Core 0.6
retired or changed is used here, and a predictive model's covariance
(`scored_model`) keeps the full storage it held before. The research project
`neurips2026/` moves with it: VisualGeometryToolkitIO 0.6,
VisualGeometryToolkitEstimation 0.6.1 and VisualGeometryToolkitEval 0.5.

Until the stack's 0.6 series is registered, the workspace takes it from a
checkout of the monorepo beside this one through a temporary `[sources]`
block, removed at the release.
