# using Pkg
# Pkg.activate("./")

using JLD2, FileIO
using RandomNumbers.Xorshifts
using ProgressMeter
using Printf
using Base.Threads

function montecarlo(L, T; nconfs=10_000)
    rng = Xoroshiro128Plus()

    # set parameters & initialize
    nsweeps = 10^2
    measure_rate = nconfs
    β = 1.0 / T
    conf = rand(rng, [-1, 1], L, L)
    confs = Matrix{Int32}[]

    # @showprogress "Equilibrating..." for i = 1:nsweeps
    for i = 1:nsweeps
        # sweep
        for j ∈ 1:L
            for k ∈ 1:L
                # Periodic boundary condition
                ip_1 = j + 1 > L ? j + 1 - L : j + 1
                im_1 = j - 1 < 1 ? j + L - 1 : j - 1
                jp_1 = k + 1 > L ? k + 1 - L : k + 1
                jm_1 = k - 1 < 1 ? k + L - 1 : k - 1

                # Change in energy
                spin_value = conf[j, k]
                spin_neighbor_sum = conf[im_1, k] + conf[ip_1, k] + conf[j, jp_1]
                spin_neighbor_sum += conf[j, jm_1]

                ΔE = 2.0 * spin_value * spin_neighbor_sum
                # Metropolis criteria
                if (ΔE <= 0) || (rand(rng) < exp(-β * ΔE))
                    conf[j, k] *= -1 # flip sign
                end
            end
        end
    end

    # walk over the lattice and propose to flip each spin `nsweeps` times
    # @showprogress "Sampling..." for i = 1:nsweeps
    for i = 1:nsweeps
        for j ∈ 1:L
            for k ∈ 1:L
                # Periodic boundary condition
                ip_1 = j + 1 > L ? j + 1 - L : j + 1
                im_1 = j - 1 < 1 ? j + L - 1 : j - 1
                jp_1 = k + 1 > L ? k + 1 - L : k + 1
                jm_1 = k - 1 < 1 ? k + L - 1 : k - 1

                # Change in energy
                spin_value = conf[j, k]
                spin_neighbor_sum = conf[im_1, k] + conf[ip_1, k] + conf[j, jp_1]
                spin_neighbor_sum += conf[j, jm_1]

                ΔE = 2.0 * spin_value * spin_neighbor_sum
                # Metropolis criteria
                if (ΔE <= 0) || (rand(rng) < exp(-β * ΔE))
                    conf[j, k] *= -1 # flip sign
                end
            end
        end

        # store the spin configuration
        if iszero(mod(i, measure_rate))
            push!(confs, copy(conf))
        end
    end

    return confs
end

function ising_threaded(nconfs, L)
    Ts = 1.2:0.1:3.4

    @threads for t in Ts
        println("T = $t")
        flush(stdout)
        c = montecarlo(L, t; nconfs=nconfs)
        confs = cat(c..., dims=3)
        fileising = @sprintf "ising_%.2f.jld2" t
        @save joinpath("data", fileising) confs
        println("Done.")
    end
end

function ising_simple(nconfs)
    t = 1.2
    println("T = $t")
    flush(stdout)
    c = montecarlo(8, t; nconfs=nconfs)
    confs = cat(c..., dims=3)
    fileising = @sprintf "ising_%.2f.jld2" t
    @save joinpath("data", fileising) confs
    println("Done.")
end

# ising_threaded()
# ising_simple()
