module Deform

using JuLIP
using JuLIP.Potentials
using Main.Fast_Hertz
using Main.Minimise_Energy
using Main.Granular

import JuLIP: energy, forces, hessian_pos

export deform_Config

function deform_Config(at::AbstractAtoms, deformType::AbstractString, deformValue::AbstractFloat)
    Lx = get_data(at, "Length")
    Ly = get_data(at, "Height")
    N = get_data(at, "Num")
    ratio = get_data(at, "ratio")
    x, y, _ = xyz(at)
    r = at.M
    if deformType == "phi"
        ϕ = deformValue
        set_data!(at, "Phi", ϕ)
        ρ = sqrt(2.0*ϕ*Lx*Ly/N/π/(1 + ratio^2))
        at.M = ρ*(0.4*at.Z + 0.6)
    elseif deformType == "grow"
        grow_ratio = deformValue
        at.M = at.M*grow_ratio
        ϕ = π*sum(at.M.^2)/Lx/Ly
        set_data!(at, "Phi", ϕ)
    elseif deformType == "strain"
        γ = deformValue
        strainType = get_data(at, "StrainType")
        strain = get_data(at, "Strain")
        strain = strain + γ
        set_data!(at, "Strain", strain)
        if strainType == "SimpleShear"
            x = x + γ*y
            mycell = [Lx strain*Ly 0.0; 0.0 Ly 0.0; 0.0 0.0 1.0]
        elseif strainType == "PureShear"
            γ_temp = γ/(1 + strain)
            Lx = Lx * (1 + γ_temp)
            Ly = Ly / (1 + γ_temp)
            set_data!(at, "Length", Lx)
            set_data!(at, "Height", Ly)
            x = x * (1 + γ_temp)
            y = y / (1 + γ_temp)
        end
        set_cell!(at, mycell)
        X = vecs([x; y; zeros(N)'])
        at.X = X
    end

    return Main.Minimise_Energy.min_Energy(at)
end

end
