# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Statistics
using Test

import Random
Random.seed!(42)

set_n_classes(teacher_type, n_samples) = teacher_type <: SpikesInput ?
                                         n_samples :
                                         (min(4, typemax(teacher_type)) + 1)
function test_convergence(; tmp = Tempotron(10), teacher_type = Bool, T = 500,
                          method = :∇, opt = SGD(1e-3, momentum = 0.99), ν = 3,
                          n_samples = 10,
                          n_classes = set_n_classes(teacher_type, n_samples),
                          n_epochs = 20000 ÷ n_samples)

    N = length(tmp.w)

    # Generate input samples
    base_samples = [poisson_spikes_input(N, ν = ν, T = T) for j = 1:n_classes]
    samples = [(x = spikes_jitter(base_samples[n_classes * (j - 1) ÷ n_samples + 1]),
                y = teacher_type <: SpikesInput ?
                    poisson_spikes_input(1, ν = ν, T = T) :
                    min(n_classes * (j - 1) ÷ n_samples, typemax(teacher_type)))
               for j = 1:n_samples]

    # Train the tempotron
    train!(tmp, samples, epochs = n_epochs, optimizer = opt, method = method)

    # Get the tempotron's output after training
    return [teacher_type <: SpikesInput ?
            vp_distance(tmp(s.x).spikes, s.y[1], τ_q = tmp.τₘ) :
            abs(min(length(tmp(s.x).spikes), typemax(teacher_type)) - s.y)
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
        err = mean([mean(test_convergence(method = :corr, opt = SGD(0.1)))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test Multi-spike
    let target = 0.2, err
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

    # Test Chronotron
    let target = 1, err
        err = mean([mean(test_convergence(teacher_type = SpikesInput, ν = 5,
                                          tmp = Tempotron(100), n_epochs = 1000,
                                          opt = SGD(0.01))) for i = 1:n_repeats])
        @test err ≤ target
    end

    # Test ReSuMe
    let target = 2, err
        err = mean([mean(test_convergence(teacher_type = SpikesInput,
                                          method = :corr, ν = 5,
                                          tmp = Tempotron(100), n_epochs = 1000,
                                          opt = SGD(0.03)))
                    for i = 1:n_repeats])
        @test err ≤ target
    end

end
