module Experiment

using JuLIP: Atoms, AbstractCalculator, neighbourlist,
             get_data, set_data!, has_data, JVecF, mat
import JuLIP: energy, forces

struct FastLJ <: AbstractCalculator
    rcut::Float64        # cut-off radius
    rcut2::Float64       # rcut^2
    r0::Float64          # spatial scale
    fcut::Float64        # ϕlj(rcut)
    dfcut::Float64       # ∂ϕlj(rcut) / ∂(rcut^2)
    rbuf::Float64   # buffer for the neighbourlist
end

FastLJ(; r0=1.0, rcut=1.01*r0, rbuf=0.5*rcut) =
   FastLJ( rcut,
           rcut^2,
           r0,
           (r0/rcut)^12 - 2 * (r0/rcut)^6,
           6 / r0^2 * (-(r0/rcut)^14 + (r0/rcut)^8),
           rbuf )

function update!(V::FastLJ, at::Atoms{Float64, Int})
    # first check whether we already have a neighbourlist that is still usable
    if has_data(at, :nlist_fastlj_X)
        Xold = get_data(at, :nlist_fastlj_X)::Vector{JVecF}
        d2 = 0.0
        for n = 1:length(Xold)
            d2 = max(d2, sum(abs2, at.X[n] - Xold[n]))
        end
        if d2 < V.rbuf^2
            i = get_data(at, :nlist_fastlj_i)::Vector{Int}
            j = get_data(at, :nlist_fastlj_j)::Vector{Int}
            return i, j
        end
    end
    # if not, then assemble a new neighbourlist
    nlist = neighbourlist(at, V.rcut)
    set_data!(at, :nlist_fastlj_X, copy(at.X))
    set_data!(at, :nlist_fastlj_i, nlist.i)
    set_data!(at, :nlist_fastlj_j, nlist.j)
    return nlist.i, nlist.j
end

function energy(V::FastLJ, at::Atoms{Float64, Int})
    i, j = update!(V, at)
    return energy_inner(V, at.X, i, j)
end

function energy_inner(V::FastLJ, X, i, j)
    E = 0.0
    for n = 1:length(i)
        @inbounds r2 = sum(abs2, X[i[n]]-X[j[n]])
        # evaluate LJ
        r2inv = (V.r0*V.r0)/r2
        r4inv = r2inv * r2inv
        r6inv = r2inv * r4inv
        r12inv = r6inv * r6inv
        lj1 = r12inv - 2 * r6inv
        # evaluate a cutoff
        lj2 = lj1 - V.fcut - V.dfcut * (r2 - V.rcut2)
        lj3 = lj2 * (r2 < V.rcut2)
        E += 0.5 * lj3
    end
    return E
end

forces(V::FastLJ, at::Atoms{Float64, Int}) =
    forces!(zeros(JVecF, length(at)), V, at)

function forces!(F, V::FastLJ, at::Atoms{Float64, Int})
    i, j = update!(V, at)
    return forces_inner!(F, V, at.X, i, j)
end

function forces_inner!(F, V::FastLJ, X, i, j)
    @assert length(F) == length(X)
    r02inv = 1.0 / V.r0^2
    for n = 1:length(i)
        R = X[i[n]]-X[j[n]]
        r2 = sum(abs2, R)
        r = sqrt(r2)
        # evaluate LJ
        r2inv = (V.r0*V.r0)/r2
        r4inv = r2inv * r2inv
        r6inv = r2inv * r4inv
        r8inv = r4inv * r4inv
        r14inv = r8inv * r6inv
        dlj1 = (6 * r02inv) * ( - r14inv + r8inv)
        # evaluate a cutoff
        dlj2 = (dlj1 - V.dfcut) * (r2 < V.rcut2)
        F[i[n]] -= dlj2 * R
        F[j[n]] += dlj2 * R
    end
    return F
end

end