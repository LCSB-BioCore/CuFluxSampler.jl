using Documenter, CuFluxSampler

makedocs(
    modules = [CuFluxSampler],
    clean = false,
    format = Documenter.HTML(),
    sitename = "CuFluxSampler.jl",
    linkcheck = false,
    pages = ["README" => "index.md"; "Reference" => "reference.md"],
    strict = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/LCSB-BioCore/CuFluxSampler.jl.git",
    target = "build",
    branch = "gh-pages",
    push_preview = false,
)
