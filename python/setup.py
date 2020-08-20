# Configure PyJulia
import julia
julia.install()               # install PyCall.jl etc.

# Add the Tempotrons.jl package
from julia import Pkg
Pkg.add(url = "https://github.com/bci4cpl/Tempotrons.jl")
