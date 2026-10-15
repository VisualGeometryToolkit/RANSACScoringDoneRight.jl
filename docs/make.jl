using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)
using DocumenterCodeBlocks    # CodeBlocks(): line numbers, docstring links and tooltips in code blocks
using RansacScoringDoneRight

makedocs(;
    modules=[RansacScoringDoneRight],
    authors="James Pritts <jbpritts@gmail.com>, Felix Seegräber, Kevin Köser",
    sitename="RANSAC Scoring Done Right",
    format=Documenter.HTML(;
        assets=["assets/landing_tagline.css", "assets/landing_tagline.js", "assets/pseudocode_render.js"],
        prettyurls=get(ENV, "CI", "false") == "true" || get(ENV, "RSDR_DOCS_DEPLOY", "0") == "1",
        canonical="https://visualgeometrytoolkit.github.io/RansacScoringDoneRight.jl/",
        edit_link="main",
    ),
    plugins=[LandingPage(), CodeBlocks()],
    warnonly=[:missing_docs, :cross_references, :docs_block],
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Guides" => [
            "Experiments" => "guide/experiments.md",
            "Conventions" => "conventions.md",
        ],
        "API Reference" => [
            "Scoring" => "api/scoring.md",
            "Posterior weighting" => "api/posterior.md",
            "Pools, grids and tuning" => "api/pools.md",
            "Internals" => "api/internals.md",
        ],
    ],
)

# The site is this repository's project page
# (https://visualgeometrytoolkit.github.io/RansacScoringDoneRight.jl/):
# unversioned, at the root of the `gh-pages` branch. CI deploys it
# (.github/workflows/docs.yml) with DOCUMENTER_KEY, the private half of a
# write-enabled deploy key of this repository.
deploydocs(;
    repo="github.com/VisualGeometryToolkit/RansacScoringDoneRight.jl.git",
    branch="gh-pages",
    devbranch="main",
    versions=nothing,
)
