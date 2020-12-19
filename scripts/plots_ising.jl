using JLD2, FileIO
using Plots
using Printf
gr()

ising_magn = Vector{Float64}(undef, 20)
rbm_magn = Vector{Float64}(undef, 20)
JLD2.@load joinpath("results", "ising_magnetization.jld2") ising_magn
JLD2.@load joinpath("results", "rbm_magnetization.jld2") rbm_magn

deviations = abs.(ising_magn .- rbm_magn)
display(deviations)

Ts = LinRange(1.2, 3.4, 20)

plot(Ts, rbm_magn, label = "RBM")
plot!(Ts, ising_magn, label = "Ising")
gui()
