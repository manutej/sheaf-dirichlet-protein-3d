# sheaf-dirichlet-protein-3d

3D prototype of the **cellular-sheaf Dirichlet principle** on real protein coordinates.

This is not a claim that Navier–Stokes folds proteins. At protein Reynolds number the honest continuum limit of solvent is **Stokes flow**. The sheaf Laplacian is the discrete Dirichlet energy of the chain; Stokes/RPY is the mobility that turns that force into 3D motion.

## What it does

1. Load a PDB (RCSB) as a database row of 3D Cα configurations.
2. Build the anisotropic cellular sheaf on the contact graph. The 0-th sheaf Laplacian equals the ANM Hessian (Hu, Liu, Xia, arXiv:2501.06197).
3. Report Dirichlet energy, zero modes (global sections = rigid motions), and flexibility.
4. Optional JAX inverse-sequence: softmax logits on residue stalks, minimize burial + contact energy.
5. Stokes/Oseen field around the backbone from modal forces (Green function of 3D Stokes).

## Demo numbers (1CRN crambin, 46 residues)

- 552 contacts at 13 Å
- exactly 6 zero eigenvalues
- next eigenvalues: 0.327, 0.459, 0.691, 0.834
- E(rigid translation) ~ 0
- E(random unit displacement) ~ 7.88

## Run

```bash
pip install numpy plotly
python demo_numpy_3d.py          # writes crambin_sheaf_3d.html

pip install jax jaxlib
python jax_sheaf_fold3d.py       # spectrum + optional sequence AD
```

Full 3D incompressible NS (if you really want a grid solver around the chain):
[JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) or [dnsjax](https://github.com/gokhanyalniz/dnsjax).
Chain dynamics belong in [JAX-MD](https://github.com/jax-md/jax-md).
