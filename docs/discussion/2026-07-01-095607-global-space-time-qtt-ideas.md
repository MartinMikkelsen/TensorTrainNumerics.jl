# Global Space-Time QTT Solver: Research Directions

Context: discussion after implementing a global Crank-Nicholson space-time QTT solver and examples for 2D heat and 2D Ornstein-Uhlenbeck equations.

The 2012 Dolgov-Khoromskij-Oseledets paper gets logarithmic scaling when the full space-time solution has bounded QTT ranks. The main fragility is therefore rank growth: long intervals, complicated potentials, nonseparable coefficients, transported sharp features, and nonlinear terms can all destroy the advantage.

Promising directions:

1. Use Lindsey-style multiscale interpolative QTT construction to build better space-time initial guesses, forcing terms, coefficients, and benchmark solutions from function evaluations.
2. Add adaptive restarted block solving, where each time slab is chosen by observed rank growth, residual behavior, or temporal variation rather than fixed time chunks.
3. Extend the global system to semilinear parabolic equations through Picard or Newton iterations in space-time QTT form.
4. Compare serial, interleaved, and mixed time-space bit orderings, especially for transported or drifting solutions such as advection-diffusion and Ornstein-Uhlenbeck dynamics.
5. Use the global space-time tensor as data for extracting low-dimensional spectral, Koopman, or metastable dynamics.

Best near-term experiment: combine interpolative space-time guesses with adaptive restarts on heat, Ornstein-Uhlenbeck, Feynman-Kac, and a semilinear reaction-diffusion test. This directly targets the practical bottleneck seen in the current implementation: the solver is sensitive to the quality and ordering of the full space-time guess.
