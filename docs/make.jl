using Documenter, TensorTrainNumerics

makedocs(
    sitename = "TensorTrainNumerics.jl",
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
    pages = [
        "index.md",
        "theory.md",
        "examples.md",
        "resources.md",
        "API.md"
        ]
    )

deploydocs(
    repo = "github.com/MartinMikkelsen/TensorTrainNumerics.jl.git",
    target = "build",
    branch="gh-pages",
)