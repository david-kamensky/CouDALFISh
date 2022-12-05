# CouDALFISh
**Cou**pling, via **D**ynamic **A**ugmented **L**agrangian (DAL), of **F**luids with **I**mmersed (alternatively: **I**sogeometric) **Sh**ells (pronounced like "cuttlefish", an animal with an internal shell that lives immersed in fluid).  This module provides re-usable functionality for immersogeometric fluid--thin structure interaction analysis.  An isogeometric discretization of a thin shell, using [ShNAPr](https://github.com/david-kamensky/ShNAPr) is immersed into an unfitted finite element discretization of a fluid, using [FEniCS](https://fenicsproject.org/).  Examples provided in this repository use [VarMINT](https://github.com/david-kamensky/VarMINT) to define the fluid formulation, but this is not strictly necessary.  This module was written to support the following paper, submitted to a special issue on open-source software for partial differential equations:
```
@article{Kamensky2021,
title = "Open-source immersogeometric analysis of fluid--structure interaction using {FEniCS} and {tIGAr}",
journal = "Computers \& Mathematics with Applications",
volume = "81",
pages = "634--648",
year = "2021",
note = "Development and Application of Open-source Software for Problems with Numerical PDEs",
issn = "0898-1221",
author = "D. Kamensky"
}
```
It serves to illustrate advanced usage of tIGAr and FEniCS, and demonstrates that automated code generation can still be useful in development of custom applications where certain functionality is still implemented manually.

The following paper further extends CouDALFISh to handle a deforming fluid domain:
```
@article{Neighbor2022,
author={G. E. Neighbor and H. Zhao and M. Saraeian and M.-C. Hsu and D. Kamensky},
title={Leveraging code generation for transparent immersogeometric fluid--structure interaction analysis on deforming domains},
journal={Engineering with Computers},
year={2022},
month={Nov},
day={16},
issn={1435-5663}
}
```

CouDALFISh couples the fluid and structure subproblems using the DAL approach, as originally proposed (without a name) in Section 4 of

  https://doi.org/10.1016/j.cma.2014.10.040

The first paper using the name DAL was published later, [here](https://doi.org/10.1142/S0218202518500537), and provides some numerical analysis for a specific variant of the method.  (DAL is essentially orthogonal to the concept of immersogeometric analysis, and could be applied with approximate finite element geometries as well.)
