#include("Hertz.jl")
include("Granular_Module.jl")
include("Fast_Hertz.jl")
#include("Multi_HZ.jl")
#include("calculate_Pressure.jl")
#include("Min_Energy.jl")
#include("Deform_Config.jl")

using JuLIP
#using Minimise_Energy
#using Deform
#using Pressure
#using Hertz
using NeighbourLists
using Fast_Hertz
#using MultiHZ
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
set_data!(Config, :Phi, ϕ)
set_data!(Config, :Rad, r)
Config1 = deepcopy(Config)

Num_Cal = 1000
# get the calculator
α = get_data(Config, :Alpha)
k = 1.0
r = Config.M
rmax = maximum(r)
rmin = minimum(r)
r0 = 2*rmax
r1 = 2*rmin

# V = Hertz.HHZPotential(k, α, r0; rcutfact = 2.5)
# set_calculator!(Config, V)


V1 = Fast_Hertz.Fast_Hertz_Potential(get_data(Config, :Length), r0, r1, 2.0;k = 1.0, rcutfact = 2.5)
set_calculator!(Config1, V1)
E = 0.0
E1 = 0.0
# tic()
#  @simd for i = 1:Num_Cal #in range(1, Num_Cal)
#     @inbounds E = Hertz.energy(V, Config)
#       end
# toc()
dE = zeros(JVecF, length(Config))
 tic()
  @simd for i = 1:Num_Cal #in range(1, Num_Cal)
     @inbounds E1 = Fast_Hertz.energy(V1, Config1)
     dE = forces(V1, Config1)
       end
 toc()

# tic()
# for i = 1:Num_Cal
#     nlist = neighbourlist(Config, cutoff(V))
# end
# @show time2 = toc()

# tic()
# for i in range(1, Num_Cal)
#     E1 = energy(V1, Config1)
# end
# time2 = toc()