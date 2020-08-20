# import Tempotrons and start working!
from collections import namedtuple
import numpy as np
from julia import Tempotrons as trons
from julia.Tempotrons import InputGen as inpgen
from julia.Tempotrons import Optimizers as opts
import matplotlib.pyplot as plt

# Set parameters
N = 10
T = 500
dt = 1
t = np.arange(0, T, dt)
opt = opts.SGD(1e-4, momentum = 0.99)
nu = 3
n_samples = 10
n_epochs = 2000
tmp = trons.Tempotron(N)

# Generate input samples
Sample = namedtuple("Sample", "x y")
base_samples = [inpgen.poisson_spikes_input(N, ν = nu, T = T) for j in range(2)]
samples = [Sample(x = inpgen.spikes_jitter(base_samples[2*j // n_samples], σ = 5),
            y = bool(2*j // n_samples)) for j in range(n_samples)]

# Get the tempotron's output before training
out_b = tmp([s.x for s in samples], t = t)

# Train the tempotron
trons.train_b(tmp, samples, epochs = n_epochs, optimizer = opt)

# Get the tempotron's output after training
out_a = tmp([s.x for s in samples], t = t)

# Plots
plt.figure(figsize = (6.4, 9))
xl = (t.min(), t.max())
for s in range(len(samples)):

    col = "r" if samples[s].y else "b"

    # Input plot
    plt.subplot(len(samples), 2, 2*s + 1)
    for i in range(len(samples[s].x)):
        plt.scatter(samples[s].x[i], (i + 1)*np.ones(len(samples[s].x[i])),
        color = col, s = 2)
    plt.xlim(xl)
    plt.ylim(0.5, len(samples[s].x) + 0.5)
    plt.yticks(ticks = [1, len(samples[s].x)])
    plt.ylabel("Neuron #")
    if s < len(samples) - 1:
        locs, _ = plt.xticks()
        plt.xticks(ticks = locs, labels = ["" for t in locs])
    else:
        plt.xlabel("time [ms]")

    # Voltage plot
    plt.subplot(len(samples), 2, 2*s + 2)
    plt.plot(xl, tmp.θ*np.ones(2), "--", color = "k")
    plt.plot(t, out_b[s].V, "--", color = col)
    plt.plot(t, out_a[s].V, color = col)
    plt.xlim(xl)
    plt.ylim(min(out_a[s].V.min(), out_b[s].V.min()) - 0.1,
            tmp.θ*1.3 + 0.1)
    plt.yticks(ticks = [0, tmp.θ], labels = ["$V_0$", "$\\theta$"])
    plt.ylabel("V [mV]")
    if s < len(samples) - 1:
        locs, _ = plt.xticks()
        plt.xticks(ticks = locs, labels = ["" for t in locs])
    else:
        plt.xlabel("time [ms]")

plt.tight_layout()
plt.show()
