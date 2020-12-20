using Pkg
Pkg.activate("./")

# Total number of confs
freq = 10_000
total_sweeps = 10^9
nconfs = round(Int, total_sweeps / freq)
# System size
L = 8
# Generate the Ising configurations
include("ising.jl")
ising_threaded(freq, L, total_sweeps)

# Now fit the RBMs
include("fit.jl")
fit_per_temp(nconfs, L)

# We now analyse everything
include("analyse.jl")
compute_values(;nconfs=nconfs, L=L)
