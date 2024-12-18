using Documenter, TensorTrainNumerics

makedocs(
    sitename = "TensorTrainNumerics",
    source = "src"
)

deploydocs(
    repo = "github.com/MartinMikkelsen/TensorTrainNumerics.jl.git",
)