using Aqua

Aqua.test_all(
    TensorTrainNumerics;
    persistent_tasks = false,
    stale_deps = (ignore = [:InterpolativeQTT],)
)
