using Documenter, DocumenterCitations, TensorTrainNumerics

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)


makedocs(
    sitename = "TensorTrainNumerics.jl",
    format = Documenter.HTML(assets = ["assets/favicon.ico", "assets/citations.css"]),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Tensor Train Basics" => "theory.md",
            "Quantics Tensor Trains" => "qtt.md",
            "Solvers" => "solvers.md",
        ],
        "Examples" => "examples.md",
        "Advanced Examples" => "advanced_examples.md",
        "Resources" => "resources.md",
        "API Reference" => "API.md",
    ],
    plugins = [bib]
)

deploydocs(
    repo = "github.com/MartinMikkelsen/TensorTrainNumerics.jl.git",
    target = "build",
    branch = "gh-pages",
)
