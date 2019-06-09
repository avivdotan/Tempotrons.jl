# Tempotrons.jl
An implementation of The Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron[[2](#references)]. See the [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/TestMultiSpike.jl) files or the (equivalent) Jupyter Notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/MultiSpikeTempotron.ipynb) for simple use-cases.

## Progress
- [x] The Binary Tempotron is fully implemented.
- [x] The Multi-Spike Tempotron is fully implemented.

## Dependencies
* Julia >1.0.0
* [Roots.jl](https://github.com/JuliaMath/Roots.jl)
* [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) (to be removed)
* [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) (for Input Generation module)
* [Plots.jl](juliaplots.org) (for Plots module)
* [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) (for demos' plots)

## References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
