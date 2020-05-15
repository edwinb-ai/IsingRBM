using Random
using RandomNumbers.Xorshifts
using LinearAlgebra


rng = Xorshift1024Plus(21457)

mutable struct BoltzmannMachine
    W::AbstractArray
    rng::AbstractRNG
end

function init!(v, h; rng = nothing)
    w = Matrix{Float64}(undef, v, h)
    # @show w
    # Glorot initialization for weights
    low = -0.1 * sqrt(6.0 / (h + v))
    rand!(rng, w, [low, -low])

    # Insert weights for the bias units into the first row and first column.
    w = vcat(zeros(1, h), w)
    w = hcat(zeros(v+1), w)

    return BoltzmannMachine(w, rng)
end

sigmoid(x) = 1 / (1 + exp(-x))

to_one(x) = x == true ? 1 : 0

function kstep(W; k = 50, rng = nothing)
    (N, M) = size(W)
    samples = rand(rng, 1, N)
    neg_associations = Matrix{Float64}(undef, N, M)

    for i = 1:k-1
        hidden_activations = samples * W
        # @show size(hidden_activations)
        hidden_probs = sigmoid.(hidden_activations)
        # @show size(hidden_probs)
        hidden_states = hidden_probs .> rand(rng, (1, M))
        hidden_states = to_one.(hidden_states)
        hidden_states[:, 1] .= 1

        visible_activations = hidden_states * W'
        visible_probs = sigmoid.(visible_activations)
        # @show size(visible_probs)
        # @show size(visible_probs')
        # @show size(hidden_probs)
        neg_associations = visible_probs' * hidden_probs
        # @show size(neg_associations)
    end

    return neg_associations
end

function train(data, rbm; epochs = 3000, η = 0.001)
    (N, M) = size(data)

    # Add bias to data
    data_bias = hcat(ones(N), data)

    neg_associations = Matrix{Float64}(undef, N, M)

    for e = 1:epochs
        pos_hidden_activations = data_bias * rbm.W
        pos_hidden_probs = sigmoid.(pos_hidden_activations)
        # Fix the bias unit
        pos_hidden_probs[:, 1] .= 1
        pos_hidden_states = pos_hidden_probs .> rand(rbm.rng, size(pos_hidden_probs)...)
        pos_hidden_states = to_one.(pos_hidden_states)
        pos_associations = data_bias' * pos_hidden_probs
        # @show size(pos_associations)

        neg_associations = kstep(rbm.W; k = 5000, rng = rbm.rng)

        # Update weights.
        @. rbm.W += η * ((pos_associations - neg_associations) / N)
    end
end

function sample(rbm; k = 50, rng = nothing)
    (N, M) = size(rbm.W)
    samples = Array{Float64}(undef, k, N, N)
    # samples = Matrix{Float64}(undef, k, N)
    samples[1, :, :] = rand(rng, 1, N, N)

    for i = 1:k-1
        hidden_activations = samples[i, :, :] * rbm.W
        hidden_probs = sigmoid.(hidden_activations)
        hidden_probs[:, 1] .= 1
        hidden_states = hidden_probs .> rand(rng, 1, M)
        hidden_states = to_one.(hidden_states)

        visible_activations = hidden_states * rbm.W'
        # @show size(visible_activations)
        visible_probs = sigmoid.(visible_activations)
        samples[i+1, :, :] = visible_probs
    end

    return samples
end

data = [
    [1, 1, 1, 0, 0, 0];
    [1, 0, 1, 0, 0, 0];
    [1, 1, 1, 0, 0, 0];
    [0, 0, 1, 1, 1, 0];
    [0, 0, 1, 1, 0, 0];
    [0, 0, 1, 1, 1, 0]
]
data = reshape(data, (6, 6))

rbm = init!(6, 2; rng = rng)

train(data, rbm)

new_data = sample(rbm; k = 2000, rng = rng)
display(new_data[end, :, :])
