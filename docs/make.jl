using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)
using DocumenterCodeBlocks    # CodeBlocks(): line numbers, docstring links and tooltips in code blocks
using DocumenterInterLinks    # InterLinks(): `[x](@extref VisualGeometryToolkitCore.x)` against Core's inventory
using RANSACScoringDoneRight

# VisualGeometryToolkitCore's names (the scoring contract, the mixture toolkit)
# and VisualGeometryToolkitEstimation's (SIMFIT++) are documented in their manuals: `@extref` resolves them against the inventory
# (objects.inv) its published `dev` docs carry, as the stack's other docs do.
const LINKS = InterLinks(
    "VisualGeometryToolkitCore" => "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitCore/dev/",
    "VisualGeometryToolkitEstimation" => "https://visualgeometrytoolkit.github.io/VisualGeometryToolkitEstimation/dev/",
)

makedocs(;
    modules=[RANSACScoringDoneRight],
    authors="James Pritts <jbpritts@gmail.com>, Felix Seegräber, Kevin Köser",
    sitename="RANSAC Scoring Done Right",
    format=Documenter.HTML(;
        assets=["assets/landing_tagline.css", "assets/landing_tagline.js", "assets/pseudocode_render.js"],
        prettyurls=get(ENV, "CI", "false") == "true" || get(ENV, "RSDR_DOCS_DEPLOY", "0") == "1",
        canonical="https://ransacscoringdoneright.github.io/",
        edit_link="main",
    ),
    plugins=[LandingPage(), CodeBlocks(), LINKS],
    # The docs gates (General review, #171997, #173031, #173032): every public
    # docstring is in the manual, the doctests run, every external link
    # resolves, and nothing is warnonly — a warning fails the build. That
    # every public name HAS a docstring is test/docs_tests.jl's job (the
    # missing-docs check only sees docstrings that exist).
    checkdocs=:public,
    doctest=true,
    linkcheck=true,
    # github.com rate-limits a crawler (429 on VisualGeometryToolkitCore's
    # build, 2026-11-04), so a failure there says nothing about the link.
    linkcheck_ignore=[r"^https://github\.com/"],
    pages=[
        "Home" => "index.md",
        "Overview" => "overview.md",
        "Guides" => [
            "Experiments" => "guide/experiments.md",
            "Conventions" => "conventions.md",
            "Working on the stack" => "guide/stack.md",
        ],
        "API Reference" => [
            "Scoring" => "api/scoring.md",
            "Posterior weighting" => "api/posterior.md",
            "Pools, grids and tuning" => "api/pools.md",
            "Internals" => "api/internals.md",
        ],
    ],
)

# The site is the ransacscoringdoneright organization page (https://ransacscoringdoneright.github.io):
# unversioned, at the root of that repository's `main` branch. CI deploys it
# (.github/workflows/docs.yml) with DOCUMENTER_KEY, the private half of a
# write-enabled deploy key of that repository. Documenter deploys only when the
# building repository's name occurs in the target URL; the site is another
# repository, so the deploy config names it as the target.
const SITE_REPO = "ransacscoringdoneright/ransacscoringdoneright.github.io"
deploydocs(;
    repo="github.com/$(SITE_REPO).git",
    branch="main",
    devbranch="main",
    versions=nothing,
    deploy_config=Documenter.GitHubActions(SITE_REPO, get(ENV, "GITHUB_EVENT_NAME", ""), get(ENV, "GITHUB_REF", "")),
)
