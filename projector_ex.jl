using ITensors, ITensorMPS
using Plots, LaTeXStrings, LinearAlgebra

include("hamiltonian.jl")
include("tdvp.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 2
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)

init = [1, 2, 3, 4]
init = init/norm(init)

M_init = MPS(init, sites, maxdim = 1)
orthogonalize!(M_init, 1)

B2 = Array(M_init[2], inds(M_init[2]))
right_proj = B2*B2'

P1 = kron([1 0; 0 1], right_proj)

#Set initial time, final time, and number of steps
t0 = 0.0
T = 20
steps = 1
step_size = (T - t0)/steps

#Run tdvp 
M_n, population = tdvp_constant(H, M_init, t0, T, steps)