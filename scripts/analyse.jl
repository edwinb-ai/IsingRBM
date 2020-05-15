using Pkg
Pkg.activate("./")
using Boltzmann
using JLD2, FileIO
using Random
using RandomNumbers.Xorshifts
using Plots
using Printf
using ProgressMeter
pyplot()

to_one(x) = x == true ? 1 : 0
to_neg(x) = x == 0 ? 1 : -1

magnetization(x) = abs(sum(x) / length(x))

function load_generate(T; L = 7, n_gibbs = 1_000, seed = nothing, n_samples = 1)
    # * Create a RNG
    if isnothing(seed)
        rng = Xoroshiro128Plus()
    else
        rng = Xoroshiro128Plus(seed)
    end
    # * Some constants
    L2 = L^2

    # * Allocate to save samples
    samples = Array{Float64}(undef, L, L, n_samples)

    # * Load the respective model for the temperature
    rbm = BernoulliRBM(L^2, 32)
    model_name = @sprintf "model_%.2f.jld2" T
    JLD2.@load joinpath("models", model_name) rbm

    # * Create a random state
    confs = Array{Float64}(undef, L, L, 10000)
    fileising = @sprintf "ising_%.2f.jld2" T
    @load joinpath("data", fileising) confs
    random_sample = confs[:, :, rand(rng, 1:1000)]
    random_sample = reshape(random_sample, L2)

    @showprogress "Sampling..." for i = 1:n_samples
        # * Sample until equilibrium
        x_new = generate(rbm, random_sample; n_gibbs = n_gibbs)
        x_new = reshape(x_new, (L, L))

        # * Probabilistiaclly turn visible units on
        x_states = x_new .> rand(rng, Float64, (L, L))
        x_states = to_one.(x_states)

        # * Return to [-1, 1] set of values from the original Ising model
        x_states = to_neg.(x_states)

        # * Save the sample
        samples[:, :, i] = x_states
    end

    return samples
end

full_magnetization = Array{Vector{Float64}}(undef, 2, 20)
Ts = [
    1.20,
    1.32,
    1.43,
    1.55,
    1.66,
    1.78,
    1.89,
    2.01,
    2.13,
    2.24,
    2.36,
    2.47,
    2.59,
    2.71,
    2.82,
    2.94,
    3.05,
    3.17,
    3.28,
    3.40
]

L = 7
confs = Array{Float64}(undef, L, L, 10000)
ising_magn = Vector{Float64}(undef, 20)
rbm_magn = Vector{Float64}(undef, 20)
for (j, T) in enumerate(Ts)
    confs = Array{Float64}(undef, L, L, 10000)
    fileising = @sprintf "ising_%.2f.jld2" T
    JLD2.@load joinpath("data", fileising) confs

    rbm_samples = load_generate(T; n_samples = 10000)

    for i in axes(confs, 3)
        ising_magn[j] += magnetization(confs[:, :, i])
        rbm_magn[j] += magnetization(rbm_samples[:, :, i])
    end
end
ising_magn ./= size(confs, 3)
display(ising_magn)
JLD2.@save joinpath("results", "ising_magnetization.jld2") ising_magn
rbm_magn ./= size(confs, 3)
display(rbm_magn)
JLD2.@save joinpath("results", "rbm_magnetization.jld2") rbm_magn
