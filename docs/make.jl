using Documenter, DocumenterCitations, TensorTrainNumerics

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)


makedocs(
    sitename = "TensorTrainNumerics.jl",
    format = Documenter.HTML(assets = ["assets/favicon.ico", "assets.citations.css"]),
    pages = [
        "index.md",
        "theory.md",
        "examples.md",
        "resources.md",
        "API.md",
    ],
    plugins=[bib]
)

deploydocs(
    repo = "github.com/MartinMikkelsen/TensorTrainNumerics.jl.git",
    target = "build",
    branch = "gh-pages",
)