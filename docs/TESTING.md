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
- block eigensolver versus direct diagonalization;
- scoped QE input parsing and fail-closed unsupported input;
- commensurate-q/supercell helpers and a directly solvable Sternheimer system;
- public accessor and geometry ChainRules pullbacks.

The synthetic UPF is test data generated in memory. It is not a physical
pseudopotential and is never used as a scientific reference.

## Private verification

The separate private verifier treats this package as a black box. It uses only
exported APIs, independently pinned NC-UPF artifacts, held-out structures, and
Quantum ESPRESSO 7.5 observations. Its bounded `ci` profile adds one small real
He response/AD smoke calculation, while the complete response, phonon, polar,
and differentiability workflows remain in the manually provisioned `full`
profile.

Neither a high line-coverage number nor a green public suite substitutes for
the private oracle and differential-consistency gates.
