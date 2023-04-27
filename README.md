
# CuFluxSampler.jl

| Documentation |
|:---:|
| [![stable documentation](https://img.shields.io/badge/docs-stable-blue)](https://lcsb-biocore.github.io/CuFluxSampler.jl/stable) [![dev documentation](https://img.shields.io/badge/docs-dev-cyan)](https://lcsb-biocore.github.io/CuFluxSampler.jl/dev) |

Flux samplers for
[COBREXA.jl](https://github.com/LCSB-BioCore/COBREXA.jl/),
accelerated on GPUs via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

The repository contains the following modules with samplers:
- Affine-combination-directed Hit&Run (module `CuFluxSampler.AffineHR`)
- Artificially-Centered Hit&Run (module `CuFluxSampler.ACHR`)

Both modules export a specific function for running the sampler atop COBREXA.jl
`MetabolicModel` structure, typically called `sample`. See the code comments
and documentation for details.

Samplers support many options that can be turned on and off, in general:
- Number of points used for mixing the new run directions in `AffineHR` may be
  changed by `mix_points` parameter, and you can alternatively supply your own
  mixing matrix in `mix_mtx`.
- You can turn on/off the stoichiometry checks with `check_stoichiometry` and
  tune it with `epsilon` (in both `ACHR` and `AffineHR`)
- You can add tolerance bounds on stoichiometry in order to expand the feasible
  region a little to allow randomized runs to succeed; see
  `check_stoichiometry` and `direction_noise_max` parameters.
- You can set a seed for the GPU-generated random numbers using `seed`

Running the package code and tests requires a CUDA-capable GPU.

#### Acknowledgements

`CuFluxSampler.jl` was developed at the Luxembourg Centre for Systems
Biomedicine of the University of Luxembourg
([uni.lu/lcsb](https://www.uni.lu/lcsb)).
The development was supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://www.permedcoe.eu/)),
agreement no. 951773.

<img src="docs/src/assets/unilu.svg" alt="Uni.lu logo" height="64px">   <img src="docs/src/assets/lcsb.svg" alt="LCSB logo" height="64px">   <img src="docs/src/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px">
