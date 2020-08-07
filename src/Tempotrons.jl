"""
An implementation of The Binary Tempotron [1] and the Multi-Spike Tempotron[2].

# References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
"""
module Tempotrons

export Tempotron, Train!, Optimizers, GetSTS, Pretrain!

include("Optimizers.jl")
include("InputGen.jl")
include("BaseTempotron.jl")
include("CorrelationLearning.jl")
include("BinaryTempotron.jl")
include("MultiSpikeTempotron.jl")
include("Plots.jl")

end
