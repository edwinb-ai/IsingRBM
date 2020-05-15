# Ising RBMs

This repository holds the code for attempting to replicate the work
by [Torlai & Melko, 2016](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.165134)
were they use [Restricted Boltzmann Machines (RBMs)](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
to unsupervisingly train them on the Ising model and generate samples from the new RBMs.

## Methodology

The methodology followed here is not quite the same as the one reported. This will be a work in progress (WIP).

The following is a list that needs to be updated:

- [ ] Use of **contrastive divergence** instead of **persistent contrastive divergence**. This should not actually be a problem, but this should be adjusted.
- [ ] Control the actual step of the *Gibbs sampling* in the **k-step constrastive divergence** algorithm.
- [ ] Use the same size for the lattice, I used `L = 7` instead of the reported `L = 8`.
- [ ] Compute the rest of the thermodynamic observables.
- [ ] Use the same size for the dataset, which is `100000` instead of `10000`.
