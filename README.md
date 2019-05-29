# Tempotrons.jl
An implementation of The Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron[[2](#references)]. See the [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/TestMultiSpike.jl) files or the (equivalent) Jupyter Notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/MultiSpikeTempotron.ipynb) for simple use-cases.

## Progress
- [x] The Binary Tempotron is fully implemented.
- [ ] The Multi-Spike Tempotron's implementation sometimes fails to match the "extra" spike to the corresponding local voltage maximum when computing θ* numerically.

## TODO
* Fix the Multi-Spike Tempotron's implementation.
* Add comments for the Multi-Spike Tempotron's implementation.

## References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
