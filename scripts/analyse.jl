# using Pkg
# Pkg.activate("./")

using Boltzmann
using JLD2, FileIO
using Random
using RandomNumbers.Xorshifts
using Printf
using ProgressMeter

to_one(x) = x == true ? 1 : 0
to_neg(x) = x == 0 ? 1 : -1

magnetization(x) = abs(sum(x) / length(x))

function load_generate(T; L=7, n_gibbs=1_000, seed=nothing, n_samples=1)
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
    rbm = BernoulliRBM(L2, 32)
    model_name = @sprintf "model_%.2f.jld2" T
    JLD2.@load joinpath("models", model_name) rbm

    # * Create a random state
    confs = Array{Float64}(undef, L, L, n_samples)
    fileising = @sprintf "ising_%.2f.jld2" T
    @load joinpath("data", fileising) confs
    random_sample = confs[:, :, rand(rng, 1:n_samples)]
    random_sample = reshape(random_sample, L2)

    @showprogress "Sampling..." for i = 1:n_samples
        # * Sample until equilibrium
        x_new = generate(rbm, random_sample; n_gibbs=n_gibbs)
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

function compute_values(;nconfs=10_000, L=8)
    Ts = 1.2:0.1:3.4
    n = size(Ts, 1)
    full_magnetization = Array{Vector{Float64}}(undef, 2, n)
    confs = Array{Float64}(undef, L, L, nconfs)
    ising_magn = Vector{Float64}(undef, n)
    rbm_magn = Vector{Float64}(undef, n)

    for (j, T) in enumerate(Ts)
        confs = Array{Float64}(undef, L, L, nconfs)
        fileising = @sprintf "ising_%.2f.jld2" T
        JLD2.@load joinpath("data", fileising) confs

        rbm_samples = load_generate(T; L=L, n_samples=nconfs)

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
end

# compute_values()
