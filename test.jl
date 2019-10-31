include("Granular_Module.jl")
include("calculate_Pressure.jl")
include("Min_Energy.jl")
include("Deform_Config.jl")
include("Hertz.jl")
using JuLIP
using Hertz
using Minimise_Energy
using Deform
using Pressure
# =====================================
# ----------- Initializaiton ----------
# =====================================

tic()

Grain_Num = 1024
α = 2.0
ϕ_Initial = 0.839
Lx = 1.0
Ly = 1.0
ratio = 1.4

# read the initial infomation from the initial_config.txt file

Initial_Info = readdlm("initial_config.txt")
x = Initial_Info[:, 1]'
y = Initial_Info[:, 2]'
z = zeros(Grain_Num)'

grain_type = Initial_Info[:, 3] + 1
ρ = sqrt(2.0*ϕ_Initial*Lx*Ly/Grain_Num/π/(1+ratio^2))
M = ρ*(0.4*grain_type + 0.6)

X = [x; y; z]
X = vecs(X)

Config_Initial = gran.setup_cell(Grain_Num, ϕ_Initial, α)
Config_Initial.X .= X
Config_Initial.Z = grain_type
Config_Initial.M = M

while true
    deformType = "grow"
    deformValue = 1.0
    Config_Initial = Deform.deform_Config(Config_Initial, deformType, deformValue)
    V_Initial = get_data(Config_Initial, "Energy")
    if V_Initial < 1.00001 * eps()
        break
    end
end

time0 = toc()

# ===============================================================================
# ---------- Find the Jamming State Roughly by Letting the Grains Grow ----------
# ===============================================================================
# setup the growth parameter for fetching the rough jamming configuration

tic()

ϕ_Target = 0.843  # this is a volume fraction which is already known bigger than the critical volume fraction
N_Grow = 50      # this is the growth step
ϕ_Increment = (ϕ_Target - ϕ_Initial)/N_Grow
 # the growth history 
Config_Grow_Hist = Array{AbstractAtoms, 1}(N_Grow)
V_Grow_Hist = zeros(N_Grow, 1)
p_Grow_Hist = zeros(N_Grow, 1)

deformType = "phi"
Config_Temp1= deepcopy(Config_Initial)
ϕ_Temp = get_data(Config_Temp1, "Phi")

 # setup the signal for jamming state
Jamming_Signal = 0
Jamming_Index = 0                              # if you need to use a variable out of loop/control body 
Config_RoughJamming = deepcopy(Config_Initial) # you should declare it firstly 

for i = 1:N_Grow
    deformValue = ϕ_Temp + ϕ_Increment
    Config_Temp2 = Deform.deform_Config(Config_Temp1, deformType, deformValue)
    Config_Grow_Hist[i] = deepcopy(Config_Temp2)

    V_Grow_Hist[i] = get_data(Config_Temp2, "Energy")
    p_Grow_Hist[i] = trace(get_data(Config_Temp2, "Pressure"))/get_data(Config_Temp2, "Dimension")

    if (Jamming_Signal == 0) & (V_Grow_Hist[i] > 5.0e-16) & (p_Grow_Hist[i] > 1e-9)
        Config_RoughJamming = deepcopy(Config_Grow_Hist[i])
        Jamming_Index = i
        Jamming_Signal = 1
    end

    if V_Grow_Hist[i] < 5.0e-16
        Jamming_Signal = 0
    end

    Config_Temp1 = deepcopy(Config_Temp2)
    ϕ_Temp = get_data(Config_Temp1, "Phi")

end

# ==============================================================================
# ---------- Find the Jamming State Accurately Using Bisection Method ---------- 
# ==============================================================================

ϕ_Jamming = get_data(Config_RoughJamming, "Phi")
Config_Jamming = deepcopy(Config_RoughJamming)

ϕ_Temp = get_data(Config_Grow_Hist[Jamming_Index - 1], "Phi")

deformType = "phi"

# the bisection method history
V_Mid_Hist = Array{Float64, 1}(0)
p_Mid_Hist = Array{Float64, 1}(0)
ϕ_Mid_Hist = Array{Float64, 1}(0)
Config_Mid_Hist = Array{AbstractAtoms, 1}(0)

while (ϕ_Jamming - ϕ_Temp) > 5.0e-8
    ϕ_Mid = (ϕ_Jamming + ϕ_Temp)/2
    deformValue = ϕ_Mid
    Config_Mid = deform_Config(Config_Jamming, deformType, deformValue)
    V_Mid = get_data(Config_Mid, "Energy")
    p_Mid = trace(get_data(Config_Mid, "Pressure"))/get_data(Config_Mid, "Dimension")
    push!(V_Mid_Hist, V_Mid)
    push!(p_Mid_Hist, p_Mid)
    push!(ϕ_Mid_Hist, ϕ_Mid)
    push!(Config_Mid_Hist, Config_Mid)
    
    if V_Mid > 5.0e-16
        ϕ_Jamming = ϕ_Mid
        Config_Jamming = deepcopy(Config_Mid)
    else
        ϕ_Temp = ϕ_Mid
    end

end
time1 = toc()

# ==============================================================================
# ---------- Test the Scaling Law (p vs ϕ) ----------
# Pick some ϕ greater than ϕ_Jamming and calculate the pressure correspondingly
# ==============================================================================

tic()
N_ϕ = 51

ϕ_Test1 = ϕ_Jamming + 10.^linspace(-6, -2, N_ϕ)
ϕ_Test1_Hist = Array{Float64, 1}(0)

Config_Test1 = Array{AbstractAtoms, 1}(N_ϕ) # This is the first main test so ...
Config_Test1_Hist = Array{AbstractAtoms, 1}(0) 

Pressure_Test1 = Array{Float64, 1}(N_ϕ)
Pressure_Test1_Hist = Array{Float64, 1}(0)

V_Test1 = Array{Float64, 1}(N_ϕ)
V_Test1_Hist = Array{Float64, 1}(0)

deformType = "phi"

Config_Temp = deepcopy(Config_Jamming)
for i = 1:N_ϕ
    deformValue = ϕ_Test1[i]
    ϕ_diff = deformValue - get_data(Config_Temp, "Phi")
    ϕ_tol = 1.0e-4
    if ϕ_diff > ϕ_tol
        k = ceil(ϕ_diff/ϕ_tol)
        ϕ_Increment = ϕ_diff/k
        for j = 1:k
            deformValue = get_data(Config_Temp, "Phi") + ϕ_Increment
            Config_Test1[i] = deform_Config(Config_Temp, deformType, deformValue)
            push!(ϕ_Test1_Hist, deformValue)
            push!(V_Test1_Hist, get_data(Config_Temp, "Energy"))
            push!(Pressure_Test1_Hist, trace(get_data(Config_Temp, "Pressure"))/get_data(Config_Temp, "Dimension"))
            push!(Config_Test1_Hist, Config_Temp)
            Config_Temp = deepcopy(Config_Test1[i])
        end
    else
        Config_Test1[i] = deform_Config(Config_Temp, deformType, deformValue)
        push!(ϕ_Test1_Hist, deformValue)
        push!(V_Test1_Hist, get_data(Config_Temp, "Energy"))
        push!(Pressure_Test1_Hist, trace(get_data(Config_Temp, "Pressure"))/get_data(Config_Temp, "Dimension"))
        push!(Config_Test1_Hist, Config_Temp)
        Config_Temp = deepcopy(Config_Test1[i])
    end

    Pressure_Test1[i] = trace(get_data(Config_Temp, "Pressure"))/get_data(Config_Temp, "Dimension")
    V_Test1[i] = get_data(Config_Temp, "Energy")

end
time3 = toc()