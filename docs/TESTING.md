# Testing

EspressoDFT separates fast implementation tests from external scientific
verification.

## Public package tests

`Pkg.test("EspressoDFT")` runs without Quantum ESPRESSO, network access, or
private fixtures. The suite includes:

- public surface, constructors, validation, and copy isolation;
- a generated minimal NC-UPF 2.0.1 fixture and malformed-UPF rejection;
- radial quadrature, special functions, and density/projector transforms;
- LDA/PBE energy-potential and response-kernel consistency;
- reciprocal indexing, Hartree invariants, and density mixing;
- explicit versus matrix-free local/nonlocal Hamiltonian action;
- persistent FFT/Hamiltonian workspace reuse and allocation bounds;
- Anderson/Pulay history, normalization, and convergence diagnostics;
- radial-projector and reciprocal-geometry cache reuse;
- block eigensolver versus direct diagonalization;
- scoped QE input parsing and fail-closed unsupported input;
- commensurate-q/supercell helpers and a directly solvable Sternheimer system;
- public accessor and geometry ChainRules pullbacks.
- a bounded synthetic ground-state, QE-input, Gamma-response, and phonon
  workflow that checks invariants without claiming external accuracy.

Optional extension smoke tests are intentionally separate from the default
dependency-free suite:

```sh
mpiexecjl -n 2 julia --project test/optional/mpi_smoke.jl
julia --project test/optional/cuda_smoke.jl
```

The MPI test uses two k points so both ranks perform work. The CUDA test uses
an NG=81 basis, above the dense toy threshold, so passing requires actual GPU
Hamiltonian kernels rather than extension loading alone.

The synthetic UPF is test data generated in memory. It is not a physical
pseudopotential and is never used as a scientific reference.

## External verification

The separate public verifier treats this package as a black box. It uses only
exported APIs, independently pinned NC-UPF artifacts, held-out structures, and
Quantum ESPRESSO 7.5 observations. Its bounded `ci` profile adds one small real
He response/AD smoke calculation, while the complete response, phonon, polar,
and differentiability workflows remain in the manually provisioned `full`
profile.

Neither a high line-coverage number nor a green package suite substitutes for
the external oracle and differential-consistency gates.
