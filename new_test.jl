include("Multi_HZ.jl")
include("Granular_Module.jl")
include("calculate_Pressure.jl")
include("Min_Energy.jl")
include("Deform_Config.jl")
using JuLIP
using Minimise_Energy
using Deform
using Pressure

# ===============================
#            new test
# ===============================

Grain_Num = 1024
α = 2.0
ϕ = 0.842

Jamming_Info = readdlm("jamming_config.txt")

x = Jamming_Info[:,1]'
y = Jamming_Info[:,2]'
z = zeros(Grain_Num)'
X = [x; y; z]
X = vecs(X)

grain_type = Jamming_Info[:,4] + 1
r = Jamming_Info[:,3]

Config = gran.setup_cell(Grain_Num, ϕ, α)
Config.X .= X
Config.M = r
Config.Z = grain_type

ϕ = π*sum(r.^2)

set_data!(Config,"Phi",ϕ)

#deformType = "grow"
#deformValue = 1.0
#Config= Deform.deform_Config(Config, deformType, deformValue)

# =======================
#   several grow steps
# =======================

N_Grow = 10
ϕ_Grow = get_data(Config, "Phi") + 10.^linspace(-6, -4, N_Grow)

p_Grow = zeros(N_Grow)'
Config_Grow = Array{AbstractAtoms, 1}(N_Grow)

deformType = "phi"
Config_Temp = deepcopy(Config)
for i = 1:N_Grow
    deformValue = ϕ_Grow[i]
    Config_Grow[i] = Deform.deform_Config(Config_Temp, deformType, deformValue)
    Config_Temp = deepcopy(Config_Grow[i])

    p_Grow[i] = trace(get_data(Config_Temp, "Pressure"))/get_data(Config_Temp, "Dimension")
end

