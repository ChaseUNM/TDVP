using ITensorMPS, ITensors
using LinearAlgebra 

include("hamiltonian.jl")
include("tdvp.jl")
#Want to test out the capabilities of creating my own Hamiltonian MPO of the Ising Model

#Create indices explicitly that will be used to create MPO
N = 3
sites = siteinds("Qubit", N)
m1 = Index(2, tags = "Bond, 1")
m2 = Index(2, tags = "Bond, 2")

#Now create MPO by setting individual tensors 
H_MPO = MPO(N)
#Create first tensor 
H_1 = zeros(2, 2, 2)
H_1[:,:,1] = [1 0; 0 -1]
H_1[:,:,2] = [1 0; 0 1]
H_1_tensor = ITensor(H_1, sites[1]', sites[1], m1)
#Create second tensor 
H_2 = zeros(2, 2, 2, 2)
H_2[1,:,:,1] = [1 0; 0 -1]
H_2[1,:,:,2] = [0 0; 0 0]
H_2[2,:,:,1] = [0 0; 0 0]
H_2[2,:,:,2] = [1 0; 0 -1]
H_2_tensor = ITensor(H_2, m1, sites[2]', sites[2], m2)
#Create third tensor 
H_3 = zeros(2, 2, 2)
H_3[1,:,:] = [1 0; 0 1]
H_3[2,:,:] = [1 0; 0 -1]
H_3_tensor = ITensor(H_3, m2, sites[3]', sites[3])

H_MPO[1] = H_1_tensor
H_MPO[2] = H_2_tensor
H_MPO[3] = H_3_tensor

#Now create MPO using OpSum
H_MPO_OpSum = ising_mpo(N, sites)

#Create Hamiltonian Matrix and make sure they are equivalent 
H = Ising(N)
println("Difference between MPO and Matrix: ", norm(H - matrix_form(H_MPO, sites)))

#Now time evolve the MPO 
init = zeros(ComplexF64, 8)
init[1] = 1.0 + 0.0*im
M_init = MPS(init, sites)


#Evolve first with just matrix exponentiation
T = 1.0
t0 = 0.0
steps = 1
u = exp(-im*H*(T - t0))*init 

#Evolve with TDVP with custom MPO
M_evolve, _ = tdvp_constant(H_MPO, M_init, t0, T, steps)

#Evolve with TDVP with OpSum MPO 
M_evolve_v2, _ = tdvp_constant(H_MPO_OpSum, M_init, t0, T, steps)

println("Difference between exponentiation and constructed MPO evolution: ", norm(u - reconstruct_arr_v2(M_evolve)))
println("Difference between exponentiation and OpSum MPO evolution: ", norm(u - reconstruct_arr_v2(M_evolve_v2)))
println("Difference between constructed MPO evolution and OpSum MPO evolution: ", norm(reconstruct_arr_v2(M_evolve) - reconstruct_arr_v2(M_evolve_v2)))
