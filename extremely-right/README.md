# RANSAC Scoring Done Extremely Right

Follow-up to *RANSAC Scoring Done Right* (NeurIPS 2026,
[RANSACScoringDoneRight.jl](https://github.com/VisualGeometryToolkit/RANSACScoringDoneRight.jl)).

That paper marginalizes the inlier scale σ out of the RANSAC score (the
profile-marginal score). One length parameter remains: the outlier half-width
`a`. This project estimates `a` from the outliers themselves, with an
extreme-value (peaks-over-threshold) fit to the tail of the distances the
current partition calls outliers. Together with the Jeffreys prior on σ, this
gives a RANSAC score with **no parameter that has units of length**. Its
ranking of partitions is invariant to the units of the residuals.

Status: exploration, in the RANSACScoringDoneRight repository (not a
workspace member yet). `explore/` holds the first simulations and
`notes/idea.md` the working notes and results so far.
