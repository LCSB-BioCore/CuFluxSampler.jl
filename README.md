
# CuFluxSampler.jl

| Documentation |
|:---:|
| [![stable documentation](https://img.shields.io/badge/docs-stable-blue)](https://lcsb-biocore.github.io/CuFluxSampler.jl/stable) [![dev documentation](https://img.shields.io/badge/docs-dev-cyan)](https://lcsb-biocore.github.io/CuFluxSampler.jl/dev) |

Flux samplers for
[COBREXA.jl](https://github.com/LCSB-BioCore/COBREXA.jl/),
accelerated on GPUs via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

The repository contents is work in progress, but the existing code should generally work. The implemented samplers currently include:
- Affine-combination-directed Hit&Run (module `CuFluxSampler.AffineHR`)

Running the package code and tests requires a CUDA-capable GPU.

#### Acknowledgements

`CuFluxSampler.jl` was developed at the Luxembourg Centre for Systems
Biomedicine of the University of Luxembourg
([uni.lu/lcsb](https://www.uni.lu/lcsb)).
The development was supported by European Union's Horizon 2020 Programme under
PerMedCoE project ([permedcoe.eu](https://www.permedcoe.eu/)),
agreement no. 951773.

<img src="docs/src/assets/unilu.svg" alt="Uni.lu logo" height="64px">   <img src="docs/src/assets/lcsb.svg" alt="LCSB logo" height="64px">   <img src="docs/src/assets/permedcoe.svg" alt="PerMedCoE logo" height="64px">
