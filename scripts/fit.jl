@everywhere function fit_ising(T; L=7, nconfs=10_000)
    L2 = L^2
    # * Allocate space for the dataset
    # confs = Array{Float64}(undef, L, L, nconfs)
    c = Matrix{Int32}(undef, L, L)
    confs = Matrix{Int32}[]

    # * Load the datasets from their files
    @inbounds for i in 1:nconfs
        fileising = @sprintf "ising_%.2f_%d.jld2" T i
        JLD2.@load joinpath("data", fileising) c
        push!(confs, copy(c))
    end

    # * Change -1 to zeros
    broadcast!(x -> x == -1 ? 0 : 1, confs, confs)

    # * Flatten the datasets, make them a 1-D array
    newconfs = reshape(confs, (L2, nconfs))

    # for i = axes(newconfs, 2)
    #     for j in eachindex(newconfs[:, i])
    #         if newconfs[j, i] == -1
    #             newconfs[j, i] = 0
    #         end
    #     end
    # end

    # * Train the RBM for the corresponding temperature
    rbm = BernoulliRBM(L2, 32)
    fit(rbm, newconfs, n_epochs = 1000, batch_size = 50, randomize = true, lr = 0.01)

    # * Serialize and save the model
    model_name = @sprintf "model_%.2f.jld2" T
    JLD2.@save joinpath("models", model_name) rbm
end
