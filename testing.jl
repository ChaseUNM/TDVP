using ITensorMPS, ITensors
using LinearAlgebra

include("hamiltonian.jl")

N = 2 
sites = siteinds("Qubit", N)

init = [1,0,0,0]
M = MPS(init, sites)
H = xxx_mpo(N, sites, -1, 1)

H_eff = MPO(N)
H_eff[1] = H[1]
H_eff[2] = H[2]*M[2]*conj(M[2])'

println(H_eff)
H_con = contract(H_eff)
println(H_con)
println(H_con*M[1])