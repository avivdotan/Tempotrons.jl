# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Statistics
using Test

import Random
Random.seed!(42)

function test_binary(;N = 10,
                      T = 500,
                      method = :∇,
                      opt = SGD(0.01),
                      ν = 3,
                      n_samples = 10,
                      n_steps = 20000)

    tmp = Tempotron(N = N)

    # Generate input samples
    base_samples = [[PoissonProcess(ν = ν, T = T)
                     for i = 1:N]
                    for j = 1:2]
    samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                     for s ∈ base_samples[2(j-1)÷n_samples + 1]],
                y = Bool(2(j-1)÷n_samples))
               for j = 1:n_samples]

    # Train the tempotron
    for i = 1:n_steps
        s = rand(samples)
        Train!(tmp, s.x, s.y, optimizer = opt, method = method)
    end

    # Get the tempotron's output after training
    return [Bool(length(tmp(s.x).spikes) > 0) != s.y for s ∈ samples]

end

function test_multispike(;N = 10,
                          T = 500,
                          method = :∇,
                          opt = SGD(0.01),
                          ν = 3,
                          n_samples = 10,
                          n_classes = 5,
                          n_steps = 20000)

    tmp = Tempotron(N = N)

    # Generate input samples
    base_samples = [[PoissonProcess(ν = ν, T = T)
                     for i = 1:N]
                    for j = 1:n_classes]
    samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                     for s ∈ base_samples[n_classes*(j - 1)÷n_samples + 1]],
                y = n_classes*(j - 1)÷n_samples)
               for j = 1:n_samples]

    # Train the tempotron
    for i = 1:n_steps
        s = rand(samples)
        Train!(tmp, s.x, s.y, optimizer = opt, method = method)
    end

    # Get the tempotron's output after training
    return [abs(length(tmp(s.x).spikes) - s.y) for s ∈ samples]

end

# Test parameters
n_repeats = 20

# Test Binary
target = 0.03
err = mean([mean(test_binary())
            for i = 1:n_repeats])
@test err ≤ target

# Test Correlation-based Binary
target = 0.2
err = mean([mean(test_binary(method = :corr))
            for i = 1:n_repeats])
@test err ≤ target

# Test Multi-spike
target = 0.5
err = mean([mean(test_multispike())
            for i = 1:n_repeats])
@test err ≤ target

# Test Correlation-based Multi-spike
target = 1
err = mean([mean(test_multispike(method = :corr))
            for i = 1:n_repeats])
@test err ≤ target
