# Tempotrons.jl
An implementation of The Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron[[2](#references)]. See the [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/test/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/test/TestMultiSpike.jl) or the (equivalent) Jupyter Notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/MultiSpikeTempotron.ipynb) for simple use-cases.

## Installation
First, install [Julia](https://julialang.org/) (version 1.1.0 or above).

In the Julia REPL, press `]` to enter package mode then run:
```console
pkg> add https://github.com/bci4cpl/Tempotrons.jl.git
```

To run the demos, please install [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) (see installation notes to use preinstalled python etc. and make sure [matplotlib](https://matplotlib.org/) is installed at the referenced python install) to enable the PyPlot backend for [Plots.jl](http://docs.juliaplots.org/latest/):
```console
pkg> add PyCall PyPlot
```
(Alternatively, you can change the code to use a different backend.)

To run the notebooks, please install [IJulia.jl](https://github.com/JuliaLang/IJulia.jl) kernel (see installation notes to use preinstalled jupyter etc.):
```console
pkg> add IJulia
```

## References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
