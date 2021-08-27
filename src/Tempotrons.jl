"""
An implementation of the Binary Tempotron [1], Multi-Spike Tempotron[2],
Chronotron [3] and ReSuMe [4].

Use the [`Tempotron`](@ref) struct to create a tempotron, then train it using
the [`train!`](@ref) method.

# Examples:

```julia
using Tempotrons
input = InputGen.poisson_spikes_input(10, ν = 5, T = 500)
tmp = Tempotron(10)                                             # Create a tempotron
train!(tmp, input, true)                                        # Binary tempotron
train!(tmp, input, true, method = :corr)                        # Binary correlation-based
train!(tmp, input, 3)                                           # Multi-spike tempotron
train!(tmp, input, 5, method = :corr)                           # Multi-spike correlation-based
train!(tmp, input, SpikesInput([[50, 100]]))                    # Chronotron
train!(tmp, input, SpikesInput([[50, 100]]), method = :corr)    # ReSuMe
output = tmp(input).spikes                                      # Get output spikes
voltage_trace = tmp(input, t = collect(0:500)).V                # Get output voltage trace
```

# References

## Binary tempotron:

[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

## Multi-spike tempotron:

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)

## Chronotron

[3] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)

## ReSuMe

[4] [Ponulak F. and Kasiński A. (2010). Supervised Learning in Spiking Neural Networks with ReSuMe: Sequence Learning, Classification, and Spike Shifting. Neural Computation, 22(2), 467-510](https://www.mitpressjournals.org/doi/abs/10.1162/neco.2009.11-08-901)
"""
module Tempotrons

# Dependencies
using Roots
using Distributions
using Statistics
using Random
using Requires

# Exports
export InputGen, Optimizers         # submodules
export Tempotron                    # main structure
export train!, get_sts, pretrain!   # methods
export SpikesInput, TimeInterval    # additional structures
export vp_distance                  # utils

# Optimizers submodule
include("Optimizers.jl")
using ..Optimizers

# Inputs submodule
include("Inputs.jl")
using ..Inputs

# Input generation submodule
include("InputGen.jl")
using ..InputGen

# Core
include("BaseTempotron.jl")
include("BinaryTempotron.jl")
include("MultiSpikeTempotron.jl")
include("CorrelationLearning.jl")
include("Chronotron.jl")
include("ReSuMe.jl")
include("STDP.jl")

# Run at using\import time
function __init__()

    # Plots
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("Plots.jl")

    return

end

end
