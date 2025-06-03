using Documenter, DocumenterVitepress, TensorTrainNumerics

makedocs(
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/MartinMikkelsen/TensorTrainNumerics.jl", 
    ),
    sitename = "TensorTrainNumerics",
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