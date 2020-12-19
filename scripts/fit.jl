using Pkg
Pkg.activate("./")
using Printf
using JLD2, FileIO
using Boltzmann

function fit_ising(T; L = 7)
    L2 = L^2
    # * Allocate space for the dataset
    confs = Array{Float64}(undef, L, L, 10000)

    # * Load the datasets from their files
    fileising = @sprintf "ising_%.2f.jld2" T
    JLD2.@load joinpath("data", fileising) confs

    # * Flatten the datasets, make them a 1-D array
    newconfs = reshape(confs, (L2, 10000))

    # * Change -1 to zeros
    for i = axes(newconfs, 2)
        for j in eachindex(newconfs[:, i])
            if newconfs[j, i] == -1
                newconfs[j, i] = 0
            end
        end
    end

    # * Train the RBM for the corresponding temperature
    rbm = BernoulliRBM(L2, 32)
    fit(rbm, newconfs, n_epochs = 10000, batch_size = 50, randomize = true, lr = 0.01)

    # * Serialize and save the model
    model_name = @sprintf "model_%.2f.jld2" T
    JLD2.@save joinpath("models", model_name) rbm
end

# Ts = LinRange(1.2, 3.4, 20)
Ts = [1.2]

map(x -> fit_ising(x; L=8), Ts)
