# CouDALFISh
**Cou**pling, via **D**ynamic **A**ugmented **L**agrangian (DAL), of **F**luids with **I**mmersed (alternatively: **I**sogeometric) **Sh**ells (pronounced like "cuttlefish", an animal with an internal shell that lives immersed in fluid).  This module provides re-usable functionality for immersogeometric fluid--thin structure interaction analysis.  An isogeometric discretization of a thin shell, using [ShNAPr](https://github.com/david-kamensky/ShNAPr) is immersed into an unfitted finite element discretization of a fluid, using [FEniCS](https://fenicsproject.org/).  Examples provided in this repository use [VarMINT](https://github.com/david-kamensky/VarMINT) to define the fluid formulation, but this is not strictly necessary.  This module was written to support the following paper, submitted to a special issue on open-source software for partial differential equations:
```
@article{Kamensky2019,
title = "Open-source immersogeometric fluid--structure interaction analysis using {FEniCS} and {tIGAr}",
journal = "Computers \& Mathematics With Applications",
author = "D. Kamensky",
note = "Under review"
}
```
It serves to illustrate advanced usage of tIGAr and FEniCS, and demonstrates that automated code generation can still be useful in development of custom applications where certain functionality is still implemented manually.

CouDALFISh couples the fluid and structure subproblems using the DAL approach, as originally proposed (without a name) in Section 4 of

  https://doi.org/10.1016/j.cma.2014.10.040

The first paper using the name DAL was published later, [here](https://doi.org/10.1142/S0218202518500537), and provides some numerical analysis for a specific variant of the method.  (DAL is essentially orthogonal to the concept of immersogeometric analysis, and could be applied with approximate finite element geometries as well.)
