using ITensors, ITensorMPS
using Plots, LaTeXStrings, LinearAlgebra

include("hamiltonian.jl")
include("tdvp.jl")
include("vectorization.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 2
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)



F = eigen(H_mat)

U = F.vectors
eigs = F.values
# display(U[:,1])


max_eig = maximum(abs.(eig))
min_eig = minimum(abs.(eig))
display(U[1,:])
println(eigs[1])
long_period = 2*pi/min_eig 
short_period = 2*pi/max_eig

# init = [1, 0, 0, 0]
# init = init/norm(init)
init = U[:,1]

M_init = MPS(init, sites)
println(norm(reconstruct_arr(2, N, M_init, sites)))
orthogonalize!(M_init, 1)

#Get P_R
B2 = Array(M_init[2], inds(M_init[2]))
P_R = B2*B2'

IP_R = kron([1 0; 0 1], P_R)

#Set initial time, final time, and number of steps
t0 = 0.0
T = 1E-10
steps = 1
step_size = (T - t0)/steps

#Run tdvp 
M_n, population = tdvp_constant(H, M_init, t0, T, steps)

#Get P_L
A1 = Array(M_n[1], inds(M_n[1]))
P_L = A1*A1'
P_LI = kron(P_L, [1 0; 0 1])

PL_PR = kron(P_L, P_R)

exp_result = exp(IP_R - PL_PR + P_LI)

exp_result2 = kron(exp(P_L)*(I - P_L), exp(P_R)) + kron(P_L*exp(P_L), exp(0 .*P_R))

#Test commutation with Hamiltonian 
p1 = (IP_R - PL_PR + P_LI)*H_mat 
p2 = H_mat*(IP_R - PL_PR + P_LI) 

#See what the sum of the projectors is when maxdim = 1, then we can look at the error 
total_proj = IP_R - PL_PR + P_LI 
display(total_proj)

display(exp(-im*total_proj*H_mat*T))
u_n = exp(-im*H_mat*T)*init
display(u_n)
arr = reconstruct_arr(2, N, M_n, sites)
display(arr)
exact = exp(-im*eigs[1]*T)*init
display(exact)
println(norm(arr - exact))
println(norm(u_n - exact))
println(norm(arr - u_n))

init = [1,0,0,0]
M_init = MPS(init, sites, maxdim = 1)

c = U\init 
C_diag = diagm(c)
CU = U*C_diag
expEig = exp.(-im*eigs*T)
exactSol = CU*expEig

u_n = exp(-im*H_mat*T)*init

M_n, population = tdvp_constant(H, M_init, t0, T, steps)

arr = reconstruct_arr(2, N, M_n, sites)
println("-------------------------------------------------")
display(arr)
display(u_n)
display(exactSol)
println(norm(arr - exactSol))
println(norm(u_n - exactSol))
println(norm(arr - u_n))