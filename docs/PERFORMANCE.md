# Performance snapshot

This is a regression snapshot, not a claim that the two packages expose
identical functionality. DFTK is used only by the external benchmark harness;
EspressoDFT neither imports nor depends on it.

## Method

Measurements were taken on 2026-07-24 on one CPU socket of the Matrix Lab
192-core/8×H200 server, using Julia 1.12.1, one Julia thread, one BLAS thread,
one OpenMP thread, PBE, the same pinned SG15/Dojo NC Si UPF, `Ecut=10 Ha`, and
identical cells/k meshes. Reported SCF times are the second run in one Julia
process, so package compilation and setup are excluded. Both codes converge to
a density threshold of `1e-8`.

| case | EspressoDFT before | EspressoDFT current | DFTK | current / DFTK |
|---|---:|---:|---:|---:|
| Si2, Gamma | 1.468 s | 0.418 s | 0.232 s | 1.81× |
| Si2, 3×3×3 k mesh | 21.536 s | 6.096 s | 1.752 s | 3.48× |
| Si16, Gamma | >249 s, not converged in 240 steps | 20.543 s, 13 steps | 11.631 s | 1.77× |

For Si2 on the 3×3×3 mesh, current EspressoDFT allocates 3.90 GB per hot SCF
instead of 15.45 GiB before this work. Eight Julia threads reduce the current
hot time from 6.096 s to 3.710 s while preserving the energy and residual.

The converged EspressoDFT/DFTK energy differences remain 0.0104 meV/atom for
Si2 Gamma and 0.0118 meV/atom for the 3×3×3 mesh. Thus the speedup does not come
from loosening the scientific comparison.

## What changed

- Routine bases no longer build an `NG×NG` dense Hamiltonian; a restarted
  preconditioned block eigensolver applies it matrix-free.
- Hamiltonian FFT grids, FFT plans, density grids, G-to-FFT indices, kinetic
  energies, and radial projectors are reused.
- A Kerker-preconditioned Anderson/Pulay mixer reduces SCF iteration count and
  exposes bounded convergence histories.
- CPU k points and density reconstruction are threaded. Optional MPI and CUDA
  extensions provide tested distributed-k-point and GPU-Hamiltonian baselines.

## Remaining gap

The main remaining hot-path cost is the block eigensolver's projected
eigensystem, residual/SVD rank decision, and thick-restart temporary matrices.
The 3×3×3 case still allocates about five times DFTK's measured 0.80 GiB. The
next performance phase should therefore make Davidson/Ritz workspaces
persistent and replace the full correction-block SVD with an allocation-bounded
orthogonalization/rank-revealing update; changing the AD design is neither
necessary nor desirable.
