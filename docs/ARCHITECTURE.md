# Architecture

## Independence boundary

EspressoDFT is a clean, independent implementation. It does not import, wrap,
or call DFTK or Quantum ESPRESSO. The runtime dependency graph contains only
generic Julia numerical infrastructure: linear algebra, FFTs, special
functions, ChainRules, and Libxc binaries.

Quantum ESPRESSO 7.5 is confined to the private verification repository. It is
an external numerical oracle there and is neither a build dependency nor a
runtime fallback of this package.

## Minimal computational stack

The code follows the physical dependency order:

1. `upf.jl` parses the frozen norm-conserving UPF subset.
2. `radial.jl` evaluates local, core-density, atomic-density, and projector
   radial transforms.
3. `types.jl` defines immutable crystals, Kohn-Sham models, exact plane-wave
   cutoffs, full k meshes, and SCF options.
4. `xc.jl` evaluates one internally consistent LDA or PBE energy and potential.
5. `hamiltonian.jl` builds local coefficients, Ewald and nonlocal terms, and
   applies dense or matrix-free Kohn-Sham Hamiltonians. Large bases use the
   package's own restarted, preconditioned block-Davidson solver.
6. `ground_state.jl` solves the density fixed point and evaluates stationary
   energy, analytic forces, and frozen-topology stress.
7. `response.jl` evaluates atomic and commensurate-supercell phonon response.
   Its homogeneous-electric-field path solves the occupied-projected
   Sternheimer equations directly, without constructing or truncating an
   empty-band sum. The same self-consistent first-order orbitals produce both
   Born effective charges and dielectric tensors; crystal symmetry selects
   only independent field directions.
8. `chainrules.jl` defines the public differentiation boundary.

There is one canonical numerical model behind both native Julia construction
and QE-compatible input parsing.

## Differentiability is a model property

The differentiable chart contains Cartesian nuclear coordinates and the cell
matrix. Cutoffs, integer G lists, FFT sizes, k grids, occupations,
pseudopotential selection, and parser choices are discrete configuration and
are intentionally non-differentiable.

`ground_state` has a custom reverse rule. Its pullback uses stationary
derivatives—forces, stress, density response, and force constants—and never
records or differentiates an SCF/mixing/eigensolver iteration tape. Thus a
change in convergence history does not redefine the derivative. Degenerate
occupied orbitals enter through their subspace rather than through derivatives
of individually labelled eigenvectors.

All strain derivatives freeze the primal integer G lists and FFT topology.
This makes the discrete plane-wave problem piecewise differentiable and avoids
mistaking a plane wave crossing the hard cutoff for physical stress.
The same rule is used for the covariant k derivative that sources the
Sternheimer electric response: k changes continuously while the primal G lists
remain fixed. The shifted occupied subspace is parallel transported before the
derivative is formed, so the result does not depend on arbitrary occupied-band
phases or rotations.

## Verification boundary

Public tests check the exported API, validation invariants, and local numerical
identities such as PBE energy–potential consistency. The private verifier owns
the licensed test fixtures, QE 7.5 observations, held-out structures, mutation
sentinels, differentiability tests, and complete workflow tests. Passing only
the public tests is not evidence of phase completion.
