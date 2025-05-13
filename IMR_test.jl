using ITensorMPS, ITensors 
using Plots, LinearAlgebra

include("hamiltonian(5-5).jl")
include("tdvp(5-12).jl")

N = 2
sites = siteinds("Qubit", N)

Ising_Mat = xxx(N, 1, 1)
Ising_MPO = xxx_mpo(N, sites, 1, 1)

t0 = 0.0
T = 10.0
steps = 20000
h = (T - t0)/steps

init = zeros(ComplexF64, 2^N)
init[1] = 1.0+0.0*im 

init_MPS = MPS(init, sites)

stor_arr = zeros(ComplexF64, (steps + 1, 2^N))
stor_arr[1,:] = init

for i = 1:steps 
    x = exp(-im*Ising_Mat*h)*init
    init .= x
    stor_arr[i + 1,:] = init 
end 

M_final, population = tdvp_constant(Ising_MPO, init_MPS, t0, T, steps)


p1 = plot(LinRange(t0,T,steps + 1), abs2.(stor_arr))
p2 = plot(LinRange(t0,T,steps + 1), abs2.(population))
p3 = plot(LinRange(t0, T, steps + 1), abs2.(stor_arr) - abs2.(population))
display(p3)

H_mat = [-1 -1 -1 0; -1 1 0 -1; -1 0 1 -1; 0 -1 -1 -1]
M_vec = zeros(ComplexF64, 2^N)
M_vec[1] = 1.0+0.0*im 


LHS = (I + h/2*im*H_mat)
RHS = (I - h/2*im*H_mat)*M_vec

println("M evolve IMR")
display(LHS\RHS)


