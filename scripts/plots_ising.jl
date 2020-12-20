# using Pkg
# Pkg.activate("./")

using JLD2, FileIO
using Plots
using Printf
gr()

Ts = 1.2:0.1:3.4
n = size(Ts, 1)

ising_magn = Vector{Float64}(undef, n)
rbm_magn = Vector{Float64}(undef, n)
JLD2.@load joinpath("results", "ising_magnetization.jld2") ising_magn
JLD2.@load joinpath("results", "rbm_magnetization.jld2") rbm_magn

deviations = abs.(ising_magn .- rbm_magn)
display(deviations)

plot(Ts, rbm_magn, label = "RBM")
plot!(Ts, ising_magn, label = "Ising")
gui()
