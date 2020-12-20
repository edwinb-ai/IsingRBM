using Distributed

@everywhere using Pkg
@everywhere Pkg.activate("./")
@everywhere using JLD2, FileIO
@everywhere using Random
@everywhere using RandomNumbers.Xorshifts
@everywhere using ProgressMeter
@everywhere using Printf
@everywhere using Boltzmann
# using Base.Threads

# Total number of confs
freq = 1_000
total_sweeps = 10^9
nconfs = round(Int, total_sweeps / freq)
Ts = 1.2:0.1:3.4
# System size
L = 8
# Generate the Ising configurations
@everywhere include(joinpath("scripts", "ising.jl"))
pmap(x -> ising_simple(freq, L, total_sweeps, x), Ts)
# Now fit the RBMs
@everywhere include(joinpath("scripts", "fit.jl"))
pmap(x -> fit_ising(x; L=L, nconfs=nconfs))

# We now analyse everything
include("analyse.jl")
compute_values(;nconfs=nconfs, L=L)
