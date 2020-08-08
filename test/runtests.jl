# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Statistics
using Test

import Random
Random.seed!(42)

function test_binary(;tmp = Tempotron(N = 10),
                      T = 500,
                      method = :∇,
                      opt = SGD(1e-4, momentum = 0.99),
                      ν = 3,
                      n_samples = 10,
                      n_steps = 20000)

    N = length(tmp.w)

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

function test_multispike(;tmp = Tempotron(N = 10),
                          T = 500,
                          method = :∇,
                          opt = SGD(1e-4, momentum = 0.99),
                          ν = 3,
                          n_samples = 10,
                          n_classes = 5,
                          n_steps = 20000)

    N = length(tmp.w)

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

# Convergence tests
let n_repeats = 20

    # Test Binary
    let target = 0.03, err
        err = mean([mean(test_binary())
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Correlation-based Binary
    let target = 0.2, err
        err = mean([mean(test_binary(method = :corr, opt = SGD(0.01)))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Multi-spike
    let target = 0.5, err
        err = mean([mean(test_multispike())
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Correlation-based Multi-spike
    let target = 1, err
        err = mean([mean(test_multispike(method = :corr, opt = SGD(0.01)))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

end
