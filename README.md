<img src="logo/logo_transparent_banner.png" width="800">

An implementation of the Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron [[2](#references)].
See [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestMultiSpike.jl)
or the [Jupyter](https://jupyter.org/) notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/MultiSpikeTempotron.ipynb)
for simple use-cases.

For a reproduction of the toy model from [[2](#references)] see [`TestAggLabels.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestAggLabels.jl)
and [`TestAggLabelsAll.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestAggLabelsAll.jl)
or the Jupyter notebook [`AggregateLabels.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/AggregateLabels.ipynb).

An interactive [Pluto](https://github.com/fonsp/Pluto.jl) notebook demonstration of the STS [[2](#references)]
is available at [`STS.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/STS.jl)

## Installation
First, install [Julia](https://julialang.org/) (version 1.5.0 or above).

In the Julia REPL, press `]` to enter package mode then run:
```console
pkg> add https://github.com/bci4cpl/Tempotrons.jl
```

To run the demos, also install [Plots](https:/http://docs.juliaplots.org) nad, optionally, [ProgressMeter](https://github.com/timholy/ProgressMeter.jl) for progress bars.:
```console
pkg> add Plots ProgressMeter
```

To run the Jupyter notebooks, please install [IJulia](https://github.com/JuliaLang/IJulia.jl) kernel (see installation notes to use preinstalled jupyter etc.). It is also recommended to install the [PlotlyJS](https://github.com/JuliaPlots/PlotlyJS.jl) backend fror plots and its [ORCA](https://github.com/sglyon/ORCA.jl) dependency:
```console
pkg> add IJulia PlotlyJS ORCA
```

To run the Pluto notebooks, please install [Pluto](https://github.com/fonsp/Pluto.jl) and [PlutoUI](https://github.com/fonsp/PlutoUI.jl):
```console
pkg> add Pluto PlutoUI
```

## Getting started
TODO

## References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
