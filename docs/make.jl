using Documenter, TensorTrainNumerics


makedocs(
    sitename = "TensorTrainNumerics",
    source  = "src", 
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/MartinMikkelsen/TensorTrainNumerics.jl.git",
    target = "docs/build",
    branch="gh-pages",
)
