using ITensorMPS 
using ITensors 
using LinearAlgebra, Plots
gr()
include("hamiltonian.jl")
include("tdvp.jl")
N = 2
d = 4
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601]*(2*pi)
rot_freq = sum(ground_freq)/N*ones(N)
dipole = [0 0.005; 0 0]*(2*pi)
self_kerr = [0.2, 0.2]*(2*pi)
cross_kerr = zeros(2, 2)

H_s = H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)

function H_t_MPO(t)
    return piecewise_H_MPO_v2(t, pt_correct_unit, qt_correct_unit, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
end

pt = npzread("2Qubit_bspline_pt.npy")
qt = npzread("2Qubit_bspline_qt.npy")

t0 = 0
T = 300.0

pt_correct_unit = pt.*(pi/500)
qt_correct_unit = qt.*(pi/500)
steps = length(pt_correct_unit[1,:])
# steps = 1000
step_size = (T - t0)/steps

init = zeros(d^N)
init[1] = 1.0 + 0.0*im 
M_init = MPS(init, sites)
# @time begin
#     M_N, population = tdvp_time(H_t_MPO, M_init, t0, T, steps, [], true)
# end


let 
    storage_arr = zeros(ComplexF64, (d^N, steps + 1))
    init = zeros(ComplexF64, d^N)
    init[1] = 1.0 + 0.0*im 
    storage_arr[:,1] = init
    @time begin
    
        for i = 1:steps
            H_c = H_ctrl(i, pt_correct_unit, qt_correct_unit, N, d)
            H_tot = H_s + H_c
            init = exp(-im*H_tot*step_size)*init
            storage_arr[:,i + 1] = init
        end
    end
    p = plot(abs2.(storage_arr'))
    display(p)
end