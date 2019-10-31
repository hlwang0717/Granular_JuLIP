include("Granular_Module.jl")
include("Fast_Hertz.jl")
include("Min_Energy.jl")

using JuLIP
using NeighbourLists
using Main.Fast_Hertz
using Main.Minimise_Energy
using LinearAlgebra
using DelimitedFiles
################################
# get an initial configuration
################################
Grain_Num = 1024
α = 2.0
ϕ = 0.842

Jamming_Info = readdlm("jamming_config.txt")

x = Jamming_Info[:,1]'
y = Jamming_Info[:,2]'
z = zeros(Grain_Num)'
X = [x; y; z]
X = vecs(X)

grain_type = Jamming_Info[:,4] .+ 1
r = Jamming_Info[:,3]

Config = gran.setup_cell(Grain_Num, ϕ, α)
Config.X .= X
Config.M = r
Config.Z = grain_type

ϕ = π*sum(r.^2)
set_data!(Config, :Phi, ϕ)
set_data!(Config, :Rad, r)

##########################################
# minimise the potential energy
##########################################
@time Config1 = Main.Minimise_Energy.min_Energy(Config)
