module Minimise_Energy

using JuLIP
using JuLIP.Potentials
using Main.Fast_Hertz
using LinearAlgebra
#using Pressure
import JuLIP: energy, forces, hessian_pos

export min_Energy

function min_Energy(at::AbstractAtoms)
    max_iter_num = 1000

    α = get_data(at, :Alpha)
    k = 1.0

    r = at.M
    rmin = minimum(r)
    rmax = maximum(r)
    r0 = 2*rmax
    r1 = 2*rmin

    V = Fast_Hertz.Fast_Hertz_Potential(get_data(at, :Length), r0, r1, α)

    set_calculator!(at, V)

    myresult = minimise!(at, precond = :id, method = :lbfgs, gtol=1e-13, verbose=2)
    is_converged = myresult.f_converged || myresult.g_converged || myresult.x_converged
    optmsg = myresult

    iter_num = myresult.iterations
    f_calls = myresult.f_calls
    g_calls = myresult.g_calls 

    while  (myresult.iterations == max_iter_num) & !(is_converged)
        myresult = minimise!(at, precond = :id, method = :lbfgs, gtol=1e-13, verbose=2)
        iter_num += myresult.iterations
        f_calls += myresult.f_calls
        g_calls += myresult.g_calls 
        @show iter_num
    end
    optmsg = myresult
    optmsg.iterations = iter_num
    optmsg.f_calls = f_calls
    optmsg.g_calls = g_calls

    @show optmsg.iterations
    set_data!(at, :Optmsg, optmsg)
    
    Energy = energy(V,at)
    set_data!(at, :Energy, Energy)

    # p = Pressure.cal_Pressure(at)
    # set_data!(at, :Stress, p)


    return at

end

end