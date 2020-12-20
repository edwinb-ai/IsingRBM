using Pkg
Pkg.activate("./")

# Total number of confs
nconfs = round(Int, 100 / 2)
# System size
L = 8
# Generate the Ising configurations
include("ising.jl")
ising_threaded(2, L)

# Now fit the RBMs
include("fit.jl")
fit_per_temp(nconfs, L)

# We now analyse everything
include("analyse.jl")
compute_values(;nconfs=nconfs, L=L)
