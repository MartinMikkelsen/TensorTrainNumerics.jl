using Aqua

# ManifoldsBase and Manopt are loaded only via the extension / examples,
# so they never appear in Base.loaded_modules at bare `using` time.
Aqua.test_all(TensorTrainNumerics; persistent_tasks = false,
    stale_deps = (ignore = [:ManifoldsBase, :Manopt],))
