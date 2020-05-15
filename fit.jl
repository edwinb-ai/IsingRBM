using Pkg
Pkg.activate("./")
using Printf
using JLD2, FileIO
using Boltzmann

function fit_ising(T; L = 7)
    # * Allocate space for the dataset
    confs = Array{Float64}(undef, L, L, 10000)

    # * Load the datasets from their files
    fileising = @sprintf "ising_%.2f.jld2" T
    JLD2.@load joinpath("data", fileising) confs

    # * Flatten the datasets, make them a 1-D array
    newconfs = reshape(confs, (L^2, 10000))

    # * Change -1 to zeros
    for i = axes(newconfs, 2)
        for j in eachindex(newconfs[:, i])
            if newconfs[j, i] == -1
                newconfs[j, i] = 0
            end
        end
    end

    # * Train the RBM for the corresponding temperature
    rbm = BernoulliRBM(L^2, 32)
    fit(rbm, newconfs, n_epochs = 10000, batch_size = 50, randomize = true, lr = 0.01)

    # * Serialize and save the model
    model_name = @sprintf "model_%.2f.jld2" T
    JLD2.@save joinpath("models", model_name) rbm
end

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

map(fit_ising, Ts)
