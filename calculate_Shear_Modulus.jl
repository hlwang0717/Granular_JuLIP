module Shear_Modulus

using JuLIP
using Minimise_Energy
using Deform
using Pressure
using MultiHZ

export cal_Shear_Modulus

# abstract type AbstractShearInfo end

struct ShearInfo
    G_Modulus::Float64
    ShearType::String
    ShearStrain::Array{Float64,1}
    ShearStress::Array{Float64,1}
    Config_Gamma_Hist::Array{AbstractAtoms,1}
end

function cal_Shear_Modulus(at::AbstractAtoms, StrainRate::Float64, StrainStep::Integer)
    N = StrainStep
    γ = StrainRate
    ShearStress = Array{Float64,1}(N)
    ShearStrain = γ * Vector(1:N)
    Config_Gamma_Hist = Array{AbstractAtoms,1}(N)

    deformType = "strain"
    deformValue = γ
    
    Config_Temp = deepcopy(at)
    for i = 1:N
        Config_Gamma_Hist[i] = deform_Conig(Config_Temp, deformType, deformValue)
        ShearStress[i] = -get_data(Config_Temp, "Pressure")[1,2]
        Config_Temp = deepcopy(Config_Gamma_Hist[i])
    end


end

end