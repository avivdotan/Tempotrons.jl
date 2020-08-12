using Tempotrons
using Luxor
using Plots
theme(:default)
# using Random
# Random.seed!(137)

logo_file       = dirname(@__FILE__) * "\\logo.png"
line_width      = 50
width, height   = 4000, 4000
clipping        = (x = -20, y = -30, r = 1900)

# from https://github.com/JuliaLang/julia-logo-graphics
JULIA_BLUE      = RGB(Luxor.julia_blue...)   #RGB(0.251, 0.388, 0.847)
JULIA_GREEN     = RGB(Luxor.julia_green...)  #RGB(0.22, 0.596, 0.149)
JULIA_PURPLE    = RGB(Luxor.julia_purple...) #RGB(0.584, 0.345, 0.698)
JULIA_RED       = RGB(Luxor.julia_red...)    #RGB(0.796, 0.235, 0.2)

N = 10
W = 0.1ones(10)
tmp = Tempotron(N; weights = W)

t = collect(0:0.01tmp.τₛ/10:2.25tmp.τₘ)

# inp = InputGen.poisson_spikes_input(N, ν = 1000/tmp.τₘ, T = t[end])
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

tmp.w[1] = 0
tmp.w[9] = -0.2
V_purple = tmp(inp, t = t).V
tmp.w[9] = 0
tmp.w[6] = 0.15
tmp.w[2] = 0.15
V_green = tmp(inp, t = t).V
tmp.w[1] = -0.2
tmp.w[6] = 0.3
V_red = tmp(inp, t = t).V

gr(size = (width, height))
p = plot([t[begin], t[end]], tmp.θ*[1, 1], linecolor = JULIA_BLUE,
         linewidth = line_width, linestyle = :dash,
         legend = false, border = :none)
plot!(t, V_purple, linecolor = JULIA_PURPLE, linewidth = line_width)
plot!(t, V_green, linecolor = JULIA_GREEN, linewidth = line_width)
plot!(t, V_red, linecolor = JULIA_RED, linewidth = line_width)
savefig(p, logo_file)

image = readpng(logo_file)
Drawing(width, height, logo_file)
origin()
# background("grey25")
circle(clipping..., :clip)
gsave()
placeimage(image, O, centered = true)
grestore()
clipreset()
finish()
