using Pkg
Pkg.activate("./")
using JLD2, FileIO
using Random
using RandomNumbers.Xorshifts
using ProgressMeter
using Printf

# functions to obtain neighbors of a given site i
up(neighs, i) = neighs[1, i]
right(neighs, i) = neighs[2, i]
down(neighs, i) = neighs[3, i]
left(neighs, i) = neighs[4, i]

function montecarlo(L, T)
    rng = Xoroshiro128Plus()

    # set parameters & initialize
    nsweeps = 10^8
    measure_rate = 10_000
    β = 1 / T
    conf = rand(rng, [-1, 1], L, L)

    confs = Matrix{Int64}[] # storing intermediate configurations

    # build nearest neighbor lookup table
    lattice = reshape(1:L^2, (L, L))
    ups     = circshift(lattice, (-1, 0))
    rights  = circshift(lattice, (0, -1))
    downs   = circshift(lattice,(1, 0))
    lefts   = circshift(lattice,(0, 1))
    neighs = vcat(ups[:]',rights[:]',downs[:]',lefts[:]')

    @showprogress "Equilibrating..." for i = 1:nsweeps
        # sweep
        for j = eachindex(conf)
            # calculate energy difference
            ΔE = 2.0 * conf[j] * (conf[up(neighs, j)] + conf[right(neighs, j)] +
                                + conf[down(neighs, j)] + conf[left(neighs, j)])

            # Metropolis criteria
            if ΔE <= 0 || rand(rng) < exp(-β * ΔE)
                conf[j] *= -1 # flip spin
            end
        end
    end

    # walk over the lattice and propose to flip each spin `nsweeps` times
    @showprogress "Sampling..." for i = 1:nsweeps
        # sweep
        for j = eachindex(conf)
            # calculate energy difference
            ΔE = 2.0 * conf[j] * (conf[up(neighs, j)] + conf[right(neighs, j)] +
                                + conf[down(neighs, j)] + conf[left(neighs, j)])

            # Metropolis criteria
            if ΔE <= 0 || rand(rng) < exp(-β * ΔE)
                conf[j] *= -1 # flip spin
            end
        end

        # store the spin configuration
        iszero(mod(i, measure_rate)) && push!(confs, copy(conf))
    end

    return confs
end

Ts = LinRange(1.2, 3.4, 20)

function ising(T)
    println("T = $T")
    flush(stdout)
    c = montecarlo(7, T)
    confs = cat(c..., dims = 3)
    fileising = @sprintf "ising_%.2f.jld2" T
    @save joinpath("data", fileising) confs
    println("Done.")
end

map(ising, Ts)
