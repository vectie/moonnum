# MoonNum

MoonNum is the shared numerical foundation for Moonphys and MoonRobo learning.
Keep dependencies directed toward portable numeric primitives; do not import
robot products, simulator SDKs, checkpoint/pickle readers, Python or LibTorch.

- Each package owns its public concrete types; compatibility facades re-export
  these types instead of duplicating their definitions.
- Use `///|` blocks and agent-authored `.mbtx` automation.
- Preserve parameter order and document rounding, alias, rank and solver rules.
- Keep BLAS explicitly selected; do not silently select devices or numerical
  backends. Hardware and simulator execution require a separate task scope.
- Run scoped `moon check`, `moon info` and `moon fmt` after code changes.
- Write meaningful regression contracts and run relevant tests when execution is
  permitted. During the current extraction the user has deferred all tests,
  numerical samples, training, simulation and benchmarks until machines are free.
- Never describe type checking or source generation as numerical/performance
  verification. Generate `.mbti` files with `moon info`; do not edit them manually.
