module Hertz

using JuLIP
using JuLIP.Potentials: pairs
import JuLIP: energy, forces, cutoff
using NeighbourLists
using StaticArrays

const SVecG{T} = SVector{3, T}
const SMatG{T} = SMatrix{3, 3, T, 9}

export HHZPotential

@pot struct HP <: AbstractCalculator #HP means Hooke/Hertzian Potential
   V::PairPotential
   rcut::Float64
   rtol::Float64
end

# mutable struct mynlist 
#     i::Vector{Int}
#     j::Vector{Int}
#     r::Vector{Float64}
#     R::Vector{JVecF}
#     s::Vector{Float64}
# end

cutoff(V::HP) = V.rcut
"""
Contact Model 
k is the energy scale, 
here, s = r - r0, r0 is the sum of radii of two grains
α = 2  => harmonic spring model
α = 2.5 => Hertzian contact model 
"""
HHZ(k, α) = @analytic s -> k * (-s)^α # harmonic or Hertzian potential
@inline evaluate_d_HHZ(p::Shift{-1}, s) = r < p.rcut ? (@D p.V(s)) : 0.0
@inline grad_HHZ(p::PairPotential, s::T, r::T, R::JVec{T}) where {T <: Real} = (evaluate_d_HHZ(p, s) / r) * R

@inline function sum_energy_HHZ(V::PairPotential, s::Vector{T}) where {T <: Real}
    E = 0.0
    @simd for n = 1:length(s)
        @inbounds E += V(s[n])
    end
    return E
end

function HHZPotential(k, α, r0::Float64; rcutfact = 2.5)
    rcut = rcutfact * r0 # rcut is rcutfact*2*R_{max}, used when searching possible pairs
    rtol = 0.5 * rcut    # rtol is the neighbourlist recomputing tolerance
    V = HHZ(k, α) * HS(0.0)
    return HP(V, rcut, rtol)
end

function energy(V::HP, at::AbstractAtoms)
    mynlist = update_neighbourlist!(V, at)
    return 0.5*sum_energy_HHZ(V.V, get_data(at, :nlist_s))
end
# compute the maximum displacement of all particles
@inline max_dr2(X::Vector{JVecF}, X_Old::Vector{JVecF}) = 
                    maximum( [ sum( (X[i] - X_Old[i]) .* (X[i] - X_Old[i]) ) for i = 1:length(X) ] )
@inline s_r_r0(i::Vector{Int}, j::Vector{Int}, Rad::Vector{Float64}, r::Vector{Float64}) = 
                    [ r[k] - Rad[i[k]] - Rad[j[k]] for k = 1:length(i) ] 

function update_neighbourlist!(V::HP, at::AbstractAtoms)
    if has_data(at, :nlist_X)
        X_Old = get_data(at, :nlist_X)::Vector{JVecF}
        
        X = at.X::Vector{JVecF}
        max_d = max_dr2(X, X_Old)
        if max_d < V.rtol*V.rtol
            nlist_i = get_data(at, :nlist_i)::Vector{Int}
            nlist_j = get_data(at, :nlist_j)::Vector{Int}
            update_distance!(nlist_i, nlist_j, at)
            return nothing
        end
    end
    nlist = neighbourlist(at, V.rcut)
    Rad = get_data(at, :Rad)
    nlist_s = s_r_r0(nlist.i, nlist.j, Rad, nlist.r)
    set_data!(at, :nlist_X, copy(at.X))
    set_data!(at, :nlist_i, nlist.i)
    set_data!(at, :nlist_j, nlist.j)
    set_data!(at, :nlist_r, nlist.r)
    set_data!(at, :nlist_R, nlist.R)
    set_data!(at, :nlist_s, nlist_s)
#    return mynlist(nlist.i, nlist.j, nlist.r, nlist.R, nlist_s)
end

@inline wrap_to_unit(inv_bins::SMatG{T}, x::SVecG{T}) where {T <: Real} = mod.(inv_bins * x, 1)
@inline adjust_dx(bins::SMatG{T}, dx::SVecG{T}) where {T <: Real} = bins * (dx - (dx .> 0.5))
# @inline x2dx(xi::SVecG{T}, xj::SVecG{T}, inv_bins::SMatG{T}, bins::SMatG{T}) where {T <: Real} = 
#                                     adjust_dx(bins, wrap_to_unit())
"""
Note: The following function is correct only when periodic boundary condition
"""
function update_distance!(nlist_i::Vector{TI}, nlist_j::Vector{TI}, at::AbstractAtoms) where TI <: Integer
    nn = length(nlist_i)
    nlist_r = Vector{Float64}()
    nlist_R = Vector{JVecF}()
    nlist_s = Vector{Float64}()
    X = at.X
    Rad = get_data(at, :Rad)
    bins = at.cell'
    inv_bins = inv(bins)
    for k = 1:nn
 #       @inbounds dx = X[nlist_j[k]] - X[nlist_i[k]]
        i = nlist_i[k]
        j = nlist_j[k]
        xi = X[i]
        xj = X[j]
        dx = xj - xi
        # Xi = wrap_to_unit(inv_bins, xi)
        # Xj = wrap_to_unit(inv_bins, xj)
        r2 = sum(abs2, dx)
        if r2 > 0.25
            dX = wrap_to_unit(inv_bins, dx)
            dx = adjust_dx(bins, dX)
            r2 = sum(abs2, dx)
        end
        r = sqrt(r2)
        push!(nlist_r, r)
        push!(nlist_R, dx)
    end
    nlist_s = s_r_r0(nlist_i, nlist_j, Rad, nlist_r)
    set_data!(at, :nlist_r, nlist_r)
    set_data!(at, :nlist_s, nlist_s)
    set_data!(at, :nlist_R, nlist_R)
#    return mynlist(nlist_i, nlist_j, nlist_r, nlist_R, nlist_s)
end

end
