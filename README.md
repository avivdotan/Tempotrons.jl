# Tempotrons.jl
An implementation of The Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron[[2](#references)]. See the [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/test/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/test/TestMultiSpike.jl) files or the (equivalent) Jupyter Notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/MultiSpikeTempotron.ipynb) for simple use-cases.

## Installation
First, install Julia >1.1.0.
In the Julia REPL, press `]` to enter package mode then run:
```console
pkg> add https://github.com/bci4cpl/Tempotrons.jl.git
```

## References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
