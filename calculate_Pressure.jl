module Pressure

using JuLIP
using JuLIP.Potentials: pairs
using MultiHZ
import JuLIP: energy, forces, hessian_pos, cutoff

export cal_Pressure

function cal_Pressure(at::AbstractAtoms)
    dim = get_data(at, "Dimension")
    Lx = get_data(at, "Length")
    Ly = get_data(at, "Height")

    α = get_data(at, "Alpha")
    α = [α α; α α]
    k = 1.0
    k = [k k; k k]

    r = at.M
    r1 = minimum(r)
    r2 = maximum(r)
    r0 = [2*r1 r1+r2; r1+r2 2*r2]

    z = [1, 2]
    V = MultiHZ.Multi_HZ(z, k, α, r0)

    p = zeros(dim, dim)

    for (i, j, r, R) in pairs(at, cutoff(V))
        a = V.Z2idx[at.Z[i]]
        b = V.Z2idx[at.Z[j]]
        f = 0.5 * grad(V.V[a,b], r, R)

        if dim == 2
            p[1,1] += f[1]*R[1]
            p[1,2] += f[1]*R[2]
            p[2,1] += f[2]*R[1]
            p[2,2] += f[2]*R[2]
        end

    end
    
    if dim == 2
        p = -p/Lx/Ly
    end

    set_data!(at, "Pressure", p)
    return p

end
    
end
