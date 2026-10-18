# =============================================================================
# src/data_artifacts.jl — the paper's published data as Julia artifacts (lazy,
# sha256-pinned tarballs on Zenodo, bound in neurips2026/Artifacts.toml).
#
# configs/data_artifacts.toml is the catalog (what each artifact holds, its
# mount, its Zenodo record, the config of record its file list derives from).
# Its format and everything that reads it — the derived file lists, the
# author-side packer, the fetch into the artifact view — are VisualGeometryToolkitEval's
# DataArtifacts, shared with the other paper projects of the stack; this file
# only binds the catalog. NEURIPS2026_ARTIFACTS_TOML names a copy of the
# bindings with other URLs and the same hashes (a fresh-machine test).
# =============================================================================
if !@isdefined(NEURIPS2026_DATA_ARTIFACTS_LOADED)

using VisualGeometryToolkitEval.DataArtifacts: DataCatalog, load_catalog, required_artifacts, artifact_files,
    fetch_artifacts!, pack_artifacts, zenodo_url, catalog_violations

const DATA_ARTIFACTS_CATALOG = normpath(joinpath(@__DIR__, "..", "configs", "data_artifacts.toml"))

"The catalog (configs/data_artifacts.toml), bound to neurips2026/Artifacts.toml or `ENV[\"NEURIPS2026_ARTIFACTS_TOML\"]`."
artifact_catalog() = load_catalog(DATA_ARTIFACTS_CATALOG)

const NEURIPS2026_DATA_ARTIFACTS_LOADED = true
end # include guard
