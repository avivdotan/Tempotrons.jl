# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Statistics
using Test

import Random
Random.seed!(42)

function test_convergence(; tmp = Tempotron(10), teacher_type = Bool, T = 500,
                          method = :∇, opt = SGD(1e-4, momentum = 0.99), ν = 3,
                          n_samples = 10,
                          n_classes = min(4, typemax(teacher_type)) + 1,
                          n_steps = 20000)

    N = length(tmp.w)

    # Generate input samples
    base_samples = [poisson_spikes_input(N, ν = ν, T = T) for j = 1:n_classes]
    samples = [(x = spikes_jitter(base_samples[n_classes * (j - 1) ÷ n_samples + 1]),
                y = min(n_classes * (j - 1) ÷ n_samples, typemax(teacher_type)))
               for j = 1:n_samples]

    # Train the tempotron
    for i = 1:n_steps
        s = rand(samples)
        train!(tmp, s.x, s.y, optimizer = opt, method = method)
    end

    # Get the tempotron's output after training
    return [abs(min(length(tmp(s.x).spikes), typemax(teacher_type)) - s.y)
            for s ∈ samples]

end

# Convergence tests
let n_repeats = 20

    # Test Binary
    let target = 0.05, err
        err = mean([mean(test_convergence()) for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Correlation-based Binary
    let target = 0.2, err
        err = mean([mean(test_convergence(method = :corr, opt = SGD(0.01)))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Multi-spike
    let target = 0.5, err
        err = mean([mean(test_convergence(teacher_type = Int))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Correlation-based Multi-spike
    let target = 1, err
        err = mean([mean(test_convergence(teacher_type = Int, method = :corr,
                                          opt = SGD(0.01))) for i = 1:n_repeats])
        @test err ≤ target
    end

end
