# Campaign 2026-09-21

Go-like anisotropic sheaf (native contacts as restrictions). Heat flow vs RPY mobility.

| PDB | n | contacts | ker H0 | l_soft | E_noise | heat 30 | RPY 30 | HP agree |
|---|---|---|---|---|---|---|---|---|
| 1L2Y | 20 | 70 | 7 | 0.0387 | 17.79 | 0.076 (99.6%) | 14.28 (19.8%) | 77% |
| 1CRN | 46 | 552 | 6 | 0.3267 | 157.68 | 0.016 (100%) | 74.63 (52.7%) | 48% |
| 1VII | 36 | 229 | 6 | 0.1008 | 65.10 | 0.043 (99.9%) | 46.20 (29.0%) | 39% |
| 2JOF | 20 | 68 | 7 | 0.0181 | 15.28 | 0.075 (99.5%) | 12.36 (19.1%) | 69% |

Heat-only is gradient descent on a quadratic well around the known native geometry; 99% drop is expected.
RPY is the same force filtered by Stokes mobility.
Auto-sequence recovers HP class on designed cages, not amino-acid identity.
Extended-chain 1L2Y: E 8953 -> 1166, RMSD 18.8 -> 13.6 A in 80 heat steps. Linearized native restrictions are a small-displacement model.

## B-factors, cutoff, scaled Stokes, GB1 + ubiquitin

### Spectra at rc = 13 A

| PDB | n | contacts | H0 | l7 |
|---|---|---|---|---|
| 1PGB | 56 | 727 | 6 | 0.650 |
| 1UBQ | 76 | 1033 | 6 | 0.034 |

Ubiquitin tiny l7 is the floppy C-terminus (G75-G76).

### Sheaf flexibility vs X-ray B-factors

| PDB | Pearson r | p |
|---|---|---|
| 1CRN | 0.385 | 8.3e-3 |
| 1PGB | 0.503 | 7.9e-5 |
| 1UBQ | 0.644 | 3.5e-10 |

1CRN cutoff sweep:

| rc A | contacts | H0 | r(flex, B) | l7 |
|---|---|---|---|---|
| 7 | 175 | 6 | 0.685 | 0.0036 |
| 9 | 269 | 6 | 0.610 | 0.048 |
| 11 | 424 | 6 | 0.440 | 0.172 |
| 13 | 552 | 6 | 0.385 | 0.327 |
| 15 | 688 | 6 | 0.387 | 0.495 |
| 18 | 842 | 6 | 0.425 | 1.111 |

Shorter cutoffs predict B-factors better (classic GNM). H0 stays 6 once the graph is rigid.

### Scaled-dt Stokes vs heat (dt_stokes = dt_heat / M_self, M_self ~ 0.028)

From 0.4 A noise, both flows reach the native well:

| PDB | steps | E0 | heat final (RMSD A) | Stokes/RPY final (RMSD A) |
|---|---|---|---|---|
| 1L2Y | 40 | 48.2 | 0.013 (0.17) | 0.028 (0.12) |
| 1CRN | 30 | 157.7 | 0.018 (0.07) | 0.42 (0.07) |
| 1PGB | 20 | 221.2 | 0.074 (0.04) | 1.00 (0.07) |

Unscaled RPY looks slow only because self-mobility is ~0.028. Same force, same well, different metric.
