using ITensors, ITensorMPS 
using LinearAlgebra, Plots, NPZ, LaTeXStrings, DelimitedFiles
using Plots.PlotMeasures
gr()
include("hamiltonian(5-5).jl")
include("tdvp(5-19).jl")
N = 2
d = 4
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601]*(2*pi)
rot_freq = sum(ground_freq)/N*ones(N)
dipole = [0 0.005; 0 0]*(2*pi)
self_kerr = [0.2, 0.2]*(2*pi)
cross_kerr = zeros(2, 2)

om = zeros(2,2)
om[1,1] = 0.027532809972830558*2*pi
om[1,2] = -0.027532809972830558*2*pi 
om[2,1] = 0.027532809972830558*2*pi 
om[2,2] = -0.027532809972830558*2*pi

display(d^N)
H_s = H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)
H_MPO = H_MPO_manual(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)

params = reshape(readdlm("params2.dat"), 160)


t0 = 0.0
T = 300.0


bc_params = bcparams((T - t0),20, om, params)

step_list = collect(1000:1000:20000)
h_list = (T - t0)./step_list
fidelity_arr_imr = zeros(length(step_list))
fidelity_arr_exp = zeros(length(step_list))

samples = 1
total_time = 0

for i in 1:length(step_list)
    UT_IMR = zeros(ComplexF64, d^N, 2^N)
    UT_exp = zeros(ComplexF64, d^N, 2^N)
    i_l = [1, 2, 5, 6]
    for j in 1:length(i_l)
        init = zeros(ComplexF64, d^N)
        init[i_l[j]] = 1.0 + 0.0*im
        M_init = MPS(init, sites, maxdim = 4)
        _, population_IMR = tdvp_time(H_MPO, M_init, t0, T, step_list[i], bc_params, 1)
        _, population_exp = tdvp_time(H_MPO, M_init, t0, T, step_list[i], bc_params, 2)
        e_IMR = population_IMR[end,:] 
        e_exp = population_exp[end,:]
        UT_IMR[:,j] = e_IMR
        UT_exp[:,j] = e_exp
    end
    Vtg = Matrix(1.0*I, d^N, 2^N)
    Vtg[3:4, 3:4] .= 0
    Vtg[6,3] = 1.0
    Vtg[5,4] = 1.0
    gate_fidelity_imr = abs.(tr(UT_IMR'*Vtg))^N/d^N
    gate_fidelity_exp = abs.(tr(UT_exp'*Vtg))^N/d^N
    fidelity_arr_imr[i] = gate_fidelity_imr
    fidelity_arr_exp[i] = gate_fidelity_exp
end

# println("Average time: ", total_time/samples)
# println(h_list)
p = plot(h_list, fidelity_arr_imr, label = "Implicit Midpoint", xlabel = "Step size", ylabel = "Gate Fidelity", dpi = 150)
plot!(h_list, fidelity_arr_exp, label = "Exponentiation")
plot!(title = "Bond Dimension: 4")
savefig(p,"Fidelity_ComparisonBD4.png")