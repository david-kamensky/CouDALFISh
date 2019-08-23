# CouDALFISh
**Cou**pling, via **D**ynamic **A**ugmented **L**agrangian (DAL), of **F**luids with **I**mmersed (alternatively: **I**sogeometric) **Sh**ells (pronounced "cuttlefish").  This module provides re-usable functionality for immersogeometric fluid--thin structure interaction analysis.  An isogeometric discretization of a thin shell, using [ShNAPr](https://github.com/david-kamensky/ShNAPr) is immersed into an unfitted finite element discretization of a fluid, using [FEniCS](https://fenicsproject.org/).  The two subproblems are coupled using the DAL approach, as originally proposed (without a name) in Section 4 of

  https://doi.org/10.1016/j.cma.2014.10.040

The first paper using the name DAL was published later, [here](https://doi.org/10.1142/S0218202518500537), and provides some numerical analysis for a specific variant of the method.  (DAL is essentially orthogonal to the concept of immersogeometric analysis, and could be applied with approximate finite element geometries as well.)

This library is still being developed and tested.
