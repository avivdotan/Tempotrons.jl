module Tempotrons

export Tempotron, Train!

include("Optimizers.jl")
include("BaseTempotron.jl")
include("BinaryTempotron.jl")
include("MultiSpikeTempotron.jl")
include("Utils.jl")
include("Plots.jl")

end
