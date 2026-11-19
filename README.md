# MoonNum

MoonBit module `vectie/moonnum@0.1.0` for shared numerical code in MoonRobo and MoonPhys.
Source lives at [vectie/moonnum](https://github.com/vectie/moonnum); a Mooncakes
registry release remains separate from this source publication.

| Package | Purpose |
| --- | --- |
| `vectie/moonnum` | Compatible Float64 Tape/Value/MLP/Adam/Normalization/Rng, dense-layer reverse kernels and direct MLP inference |
| `vectie/moonnum/geometry` | Canonical Vec3/Basis3/Transform3 extracted from MoonMoon |
| `vectie/moonnum/tensor` | Explicit F32/F64 storage, strided views, broadcasting, reductions and batched matrix multiplication |
| `vectie/moonnum/linalg` | Scaled-pivot LU solve and column-pivoted Householder least squares |
| `vectie/moonnum/blas` | Explicit portable GEMM or optional native dynamic LP64 CBLAS loading |

See [API.md](API.md) for precise numerical, ownership, shape, checkpoint and backend contracts. Generated interfaces are checked in beside each package.

The compatibility sources originated in MoonRobo's `src/learning_numeric/{autodiff,network,adam}.mbt`. Geometry originated in MoonMoon's `src/moonphys/{vector,transform}.mbt`. Original repository sources remain preserved by the integration task. Both upstream modules declare Apache-2.0; the copied license is retained here.

Validation performed: `moon check --target all`, `moon info`, `moon fmt`, and `clang -std=c11 -Wall -Wextra -Werror -fsyntax-only -I /Users/kq/.moon/include blas/blas_stub.c`. Deprecation warnings inherited from existing derived traits and Float conversions do not prevent type checking.

The initial extraction was typechecked without execution. On 19 November 2026,
all **25 native tests passed**, including scalar/block reverse AD, recurrent
finite differences, Adam tracing/counter bounds, normalization/RNG, strided
F32/F64 tensors, QR/LU, geometry and both GEMM precisions. Native CBLAS was
explicitly selected through `MOONNUM_BLAS_LIBRARY` using macOS Accelerate.
Existing compiler deprecation warnings remain. These numerical tests do not
establish benchmark performance or cross-language training parity.
