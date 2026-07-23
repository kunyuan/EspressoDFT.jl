# EspressoDFT.jl

EspressoDFT is an independent MIT-licensed Julia implementation of a focused,
differentiable plane-wave pseudopotential DFT/DFPT workflow. Its public V0
contract is frozen in the Minos `EspressoDFT` campaign.

The implementation does **not** depend on DFTK or Quantum ESPRESSO at runtime.
The separate private verification repository pins Quantum ESPRESSO 7.5 as its
external numerical oracle.

The first milestone covers spin-unpolarized insulating crystals with
norm-conserving UPF 2.0.1 pseudopotentials, LDA/PBE, full Monkhorst-Pack meshes,
ground-state observables, atomic response, phonons, polar tensors, and
ChainRules-compatible implicit derivatives.

The implementation boundary and differentiability model are described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The public/private test boundary and the fast numerical-kernel suite are
described in [`docs/TESTING.md`](docs/TESTING.md).
