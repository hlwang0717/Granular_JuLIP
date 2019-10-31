module MultiHZ

using JuLIP
using JuLIP.Potentials: pairs
import JuLIP: energy, forces, cutoff

export Multi_HZ

@pot struct MHZ{T} <: AbstractCalculator
   Z2idx::Vector{Int}
   V::Matrix{T}
   rcut::Float64
end

cutoff(V::MHZ) = V.rcut
# k is the energy scale, r0 is the sum of the two contacted grain
Hertz(k, α, r0) = @analytic r -> k * (r - r0)^α

function Multi_HZ(Z, k, α, r0; rcutfact = 2.5)
    @assert length(Z) == length(unique(Z))
    # create a mapping from atomic numbers to indices
    Z2idx = zeros(Int, maximum(Z))
    Z2idx[Z] = 1:length(Z)
    # generate the potential
    rcut = rcutfact*maximum(r0) # rcut is 5*R_{max}, used when searching possible pairs
    V = [ Hertz(k[a,b], α[a,b], r0[a,b]) * HS(r0[a,b])
        for a = 1:length(Z), b = 1:length(Z)]
    return MHZ(Z2idx, V, rcut)
end

function energy(V::MHZ, at::Atoms)
   E = 0.0
   for (i, j, r, R) in pairs(at, cutoff(V))
      a = V.Z2idx[at.Z[i]]
      b = V.Z2idx[at.Z[j]]
      E += 0.5 * V.V[a, b](r)
   end
   return E
end

function forces(V::MHZ, at::Atoms)
   F = zeros(JVecF, length(at))
   for (i, j, r, R) in pairs(at, cutoff(V))
      a = V.Z2idx[at.Z[i]]
      b = V.Z2idx[at.Z[j]]
      f = 0.5 * grad(V.V[a,b], r, R)
      F[i] += f
      F[j] -= f
   end
   return F
end

end

