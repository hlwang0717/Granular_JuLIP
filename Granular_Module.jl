module Granular

using JuLIP
using JuLIP.Potentials
using NeighbourLists
using StatsBase
import JuLIP: energy, forces, hessian_pos, Atoms

export setup_cell

function setup_cell(N, ϕ, α; Lx = 1.0, Ly = 1.0, ratio=1.4, dim = 2)
    # create computational cell
    x = Lx * rand(N)'
    y = Ly * rand(N)'
    z = zeros(N)'
    X = [x; y; z]
    X = vecs(X)
    # create atoms
    # at = Atoms()
    # at = Atoms(X, zeros(X), ones(N), rand([1,2], N), _autocell(X),
    #         (false, false, false) )5 * eye(3), false)
    at = Atoms(:X, X[:])

    ρ = sqrt(2.0*ϕ*Lx*Ly/N/π/(1+ratio^2))
    
    idx = sample(1:N, Int(N/2); replace=false, ordered=true) # index for smaller discs
    at.Z = ones(N)
    at.Z[idx] .= 2 # Here at.Z represents the tag of the grains, 
                  # at.Z[i]=1, means the i-th grain is smaller
                  # at.Z[j]=2, means the j-th grain is bigger.
    
    M = ρ*(0.4*at.Z .+ 0.6); # Here, M does not mean masses but radius.
    at.M = M
    Rad = copy(M)
 # setup the basic information
    set_data!(at, :Num, N, Inf)
    set_data!(at, :ratio, ratio, Inf)
    set_data!(at, :Alpha, α, Inf) 
    set_data!(at, :Phi, ϕ, Inf)
    set_data!(at, :Rad, Rad, Inf)

 # setup the shear type and initialize the strain 
    StrainType = "SimpleShear"
    Strain = 0.0
    set_data!(at, :StrainType, StrainType, Inf)
    set_data!(at, :Strain, Strain, Inf)

 # initialize the pressure and energy 
    set_data!(at, :Stress, zeros(dim, dim), Inf)
    set_data!(at, :Energy, 0.0, Inf)
 
 # setup the geometry information
    set_data!(at, :Length, Lx, Inf)
    set_data!(at, :Height, Ly, Inf)
    set_data!(at, :Dimension, dim, Inf)

    if StrainType == "SimpleShear"
        mycell = [Lx 0.0 0.0; Strain*Ly Ly 0.0; 0.0 0.0 1.0] # use mycell to differentiate this from JuLIP::cell
    elseif StrainType == "PureShear"
        mycell = [Lx 0.0 0.0; 0.0 Ly 0.0; 0.0 0.0 1.0]
    end
    set_cell!(at, mycell) 
    
    # set_calculator!(at, V)
    if dim == 2
        set_constraint!(at, FixedCell2D(at))
    end
    set_pbc!(at, [true, true, true])
    
    return at
end

end

gran = Granular  
