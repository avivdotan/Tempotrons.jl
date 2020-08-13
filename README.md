# <img src="logo/logo_transparent_banner.png" width="800">

A [Julia](https://julialang.org/) implementation of the Binary Tempotron [[1](#references)] and the Multi-Spike Tempotron [[2](#references)].

See [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestMultiSpike.jl)
or the [Jupyter](https://jupyter.org/) Jupyter notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/MultiSpikeTempotron.ipynb)
for simple use-cases.

For a reproduction of the toy model from [[2](#references)] see [`TestAggLabels.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestAggLabels.jl)
or the Jupyter notebook [`AggregateLabels.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/AggregateLabels.ipynb).

An interactive [Pluto](https://github.com/fonsp/Pluto.jl) notebook demonstration of the STS [[2](#references)] function
is available at [`STS.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/STS.jl).

## Installation

First, install [Julia](https://julialang.org/) (version 1.5.0 or above).

In the Julia REPL, press `]` to enter package mode then run:

```console
pkg> add https://github.com/bci4cpl/Tempotrons.jl
```

### Examples dependencies

To run the demos, also install [Plots](https:/http://docs.juliaplots.org) and, optionally, [ProgressMeter](https://github.com/timholy/ProgressMeter.jl) for progress bars:

```console
pkg> add Plots ProgressMeter
```

### Notebooks dependencies

#### Jupyter notebooks

To run the Jupyter notebooks, please install [IJulia](https://github.com/JuliaLang/IJulia.jl) kernel (see installation notes to use preinstalled jupyter etc.). It is also recommended to install the [PlotlyJS](https://github.com/JuliaPlots/PlotlyJS.jl) backend for plots and its [ORCA](https://github.com/sglyon/ORCA.jl) dependency:

```console
pkg> add IJulia PlotlyJS ORCA
```

#### Pluto notebooks

To run the Pluto notebooks, please install [Pluto](https://github.com/fonsp/Pluto.jl) and [PlutoUI](https://github.com/fonsp/PlutoUI.jl):

```console
pkg> add Pluto PlutoUI
```

## Getting started

### The `Tempotron` type

The main type provided by this package is the `Tempotron`. To create one, you must provide the number of input neurons (for a complete list of optional parameters, see the docs), e.g.:

```julia
using Tempotrons
tmp  = Tempotron(10)
tmp2 = Tempotron(500)
tmp3 = Tempotron(10, V₀ = -70, θ = -55)
tmp4 = Tempotron(10, τₘ = 20)
```

### The `SpikesInput` type

The `Tempotron`'s input is of type `SpikesInput`. To create a new `SpikesInput`, you must provide a list of spike trains where each spike train is represents the list of spike times of the corresponding input neuron. An optional parameter is the duration of the whole input. For example:
```julia
inp = SpikesInput([[10, 11, 12, 13, 14, 15, 16],
                   [8.943975412613074, 15.807304569617331, 26.527688555672533],
                   [0.48772650867156875, 8.996332849775623],
                   [5.066872939413796],
                   [8.928252059259274, 17.53078106972171],
                   [2.578155671963216, 7.825172521022958, 11.82651548544644],
                   [16.20526605836777],
                   [23.385019802078126],
                   [24, 25],
                   [25.16755219757226]])
```

Alternatively, some variations of random inputs can be generated via the `InputGen` submodule:

```julia
inp2 = InputGen.poisson_spikes_input(10, ν = 3, T = 500)
```

To have a look at the input, just import the [Plots](https:/http://docs.juliaplots.org) package and plot the input:

```julia
using Plots
plot(inp2)
```

### Getting a tempotron's output

#### Spike times

To get the tempotron's output for a given output, use:

```julia
spk  = tmp(inp).spikes
spk2 = tmp(inp2).spikes
spk3 = tmp3(inp2).spikes
spk4 = tmp4(inp2).spikes
```

This will return a named tuple with a single field, `spikes`, containing a list of the output spike times.

#### Voltage trace

To also get the output voltage trace, just provide a time grid `t`:

```julia
t1 = 0:0.1:40
spk, V = tmp(inp, t = t1)
t2 = collect(0:500)
V2 = tmp(inp2, t = t2).V
```

This will add the output an additional field `V`, containing the voltage at the times provided by `t`.
To have a look at the input, just import the [Plots](https:/http://docs.juliaplots.org) package and plot the voltage trace:

```julia
using Plots
plot(tmp, t2, V2)
```

Alternatively, you can let the `plot` function calculate the output for you:

```julia
plot(tmp, inp2)
```

#### STS

You can also have a look at the input's STS [[2](#references)] function using `GetSTS` to det a list of critical thresholds and `plotsts` for visualization:

```julia
θ⃰ = GetSTS(tmp, inp2)

using Plots
plotsts(tmp, θ⃰)
```

Again, you can let the `plotsts` function do the calculations for you:

```julia
plotsts(tmp, inp2)
```

For further reading, see the interactive [Pluto](https://github.com/fonsp/Pluto.jl) notebook demonstration [`STS.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/STS.jl).

### Training

#### Binary Tempotron

Finally, we would like to train our tempotron. This is done using the `Train!` function. For example, to train our tempotron *not* to spike for a given input using the Binary tempotron's [[1](#references)] learning rule, we can use:

```julia
Train!(tmp, inp2, false)
```

To use the correlation-based learning rule, use:

```julia
Train!(tmp, inp2, false, method = :corr)
```

and to train the tempotron to fire use:

```julia
Train!(tmp, inp2, true)
```

#### Multi-Spike Tempotron

To use the Multi-Spike Tempotron's learning rules, simply replace the binary teacher's signal to an integer one:

```julia
Train!(tmp, inp2, 2)
Train!(tmp, inp2, 3, method = :corr)
Train!(tmp, inp2, 2)
```

#### Optimizers

The `Optimizers` submodule provides a set of popular gradient-based optimizers. To use one, simply provide it to the `Train!` function:

```julia
Train!(tmp, inp2, 2, optimizer = Optimizers.SGD(1e-4, momentum = 0.99))
Train!(tmp, inp2, true, method = :corr, optimizer = Optimizers.SGD(0.01))
Train!(tmp, inp2, 2, optimizer = Optimizers.RMSprop(0.01))
```

#### Putting it all together

Simple use-cases are provided at [`TestBinary.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestBinary.jl)/[`TestMultiSpike.jl`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/examples/TestMultiSpike.jl)
or the [Jupyter](https://jupyter.org/) Jupyter notebooks [`BinaryTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/BinaryTempotron.ipynb)/[`MultiSpikeTempotron.ipynb`](https://github.com/bci4cpl/Tempotrons.jl/blob/master/notebooks/MultiSpikeTempotron.ipynb).

## References

[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
