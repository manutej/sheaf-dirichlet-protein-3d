# sheaf-dirichlet-protein-3d

3D prototype of the cellular-sheaf Dirichlet principle on real protein coordinates.

At protein Reynolds number the honest continuum limit of solvent is Stokes flow. The sheaf Laplacian is the discrete Dirichlet energy of the chain; Stokes/RPY is the mobility that turns that force into 3D motion.

## Run

```bash
pip install numpy plotly
python demo_numpy_3d.py          # writes crambin_sheaf_3d.html

pip install jax jaxlib
python jax_sheaf_fold3d.py
```

Interactive instrument (local artifacts): `sheaf-fold-3d/index.html`
Campaign numbers: [CAMPAIGN.md](CAMPAIGN.md)

## Campaign snapshot

Six folds loaded from RCSB (1L2Y, 1CRN, 1VII, 2JOF, 1PGB, 1UBQ).
Every rigid-enough contact graph has dim H0 = 6.
Heat flow on E = 1/2 x^T L_F x returns a 0.4 A-perturbed native to E~0 and RMSD < 0.2 A.
RPY Stokes reaches the same well when dt is scaled by 1/M_self.
Sheaf-Laplacian flexibility vs X-ray B-factors: 1UBQ r=0.64, 1PGB r=0.50, 1CRN r=0.39 (r=0.69 at 7 A cutoff).
Burial-only auto-sequence recovers hydrophobic class on cages (~70%), not residue identity.

This is structure-based Dirichlet descent, not sequence-to-fold. Sequence does not yet generate the contact sheaf.
