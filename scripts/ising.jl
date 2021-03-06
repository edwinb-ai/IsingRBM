@everywhere function montecarlo!(L, T, conf, rng; nconfs=10_000)
    # set parameters & initialize
    β = 1.0 / T

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

@everywhere function ising_threaded(nconfs, L, sweeps)
    Ts = 1.2:0.1:3.4
    c = Matrix{Int32}(undef, L, L)

    # @threads for t in Ts
    rng = Xoroshiro128Plus()
    rand!(rng, c, [-1, 1])

    println("T = $t")
    @showprogress "Equilibrating..." for _ in 1:sweeps
        montecarlo!(L, t, c, rng; nconfs=nconfs)
    end

    println("Sampling...")
    flush(stdout)

    j = 0
    @inbounds for i in 1:sweeps
        montecarlo!(L, t, c, rng; nconfs=nconfs)
        if iszero(mod(i, nconfs))
            j += 1
            fileising = @sprintf "ising_%.2f_%d.jld2" t j
            @save joinpath("data", fileising) copy(c)
        end
    end
    println("Done.")
    flush(stdout)
    # end
end

@everywhere function ising_simple(nconfs, L, sweeps, t)
    # t = 1.2
    c = Matrix{Int32}(undef, L, L)

    # @threads for t in Ts
    rng = Xoroshiro128Plus()
    rand!(rng, c, [-1, 1])

    println("T = $t")
    @showprogress "Equilibrating..." for _ in 1:sweeps
        montecarlo!(L, t, c, rng; nconfs=nconfs)
    end

    println("Sampling...")
    flush(stdout)

    j = 0
    @inbounds for i in 1:sweeps
        montecarlo!(L, t, c, rng; nconfs=nconfs)
        if iszero(mod(i, nconfs))
            j += 1
            fileising = @sprintf "ising_%.2f_%d.jld2" t j
            @save joinpath("data", fileising) copy(c)
        end
    end
    println("Done.")
    flush(stdout)
end

# ising_threaded()
# ising_simple()
