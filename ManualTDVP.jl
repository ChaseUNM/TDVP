using ITensors, ITensorMPS, LinearAlgebra 

include("hamiltonian.jl")
include("tdvp.jl")

#Create manual MPO and use that for a manual 3 site TDVP 
N = 3

sites = siteinds("Qubit", 3)

H1 = zeros(3, 2, 2)
H1[1,:,:] = [1 0; 0 -1]
H1[2,:,:] = [0 1; 1 0]
H1[3,:,:] = [1 0; 0 1]

H2 = zeros(3, 3, 2, 2)
H2[1,1,:,:] = [1 0; 0 -1]
H2[2,1,:,:] = [-1 0; 0 -1]
H2[3,1,:,:] = [0 -1; -1 0]
H2[3,2,:,:] = [1 0; 0 -1]
H2[3,3,:,:] = [-1 0; 0 -1]

H3 = zeros(3, 2, 2)
H3[1,:,:] = [1 0; 0 1]
H3[2,:,:] = [1 0; 0 -1]
H3[3,:,:] = [0 1; 1 0]

b1 = Index(3, "link, l = 1")
b2 = Index(3, "link, l = 2")

H1_tens = ITensor(H1, b1, sites[1], sites[1]')
H2_tens = ITensor(H2, b1, b2, sites[2], sites[2]')
H3_tens = ITensor(H3, b2, sites[3], sites[3]')

H_x_MPO = MPO(3)

H_x_MPO[1] = H1_tens
H_x_MPO[2] = H2_tens
H_x_MPO[3] = H3_tens

H_x_mat = xxx_v2(N, 1, -1)

println("Difference between MPO and Mat: ", norm(H_x_mat - matrix_form(H_x_MPO, sites)))

#Evolution just to test that it's accurate 
init = [0,1,1,0,1,0,0,0]
init = init/norm(init)
M_init = MPS(init, sites)

T = 1.0
t0 = 0.0
steps = 1

x = exp(-im*H_x_mat*(T - t0))*init 
x_MPS, _ = tdvp_constant(H_x_MPO, M_init, t0, T, steps)

println("Difference after evolution: ", norm(x - reconstruct_arr_v2(x_MPS)))

#Now do manual evolution for TDVP 
orthogonalize!(M_init, 1)
H1_eff = effective_Hamiltonian(H_x_MPO, M_init, 1)
println(H1_eff)
# H1_eff_mat, M1_vec = conversion(H1_eff, M_init[1])
# M1_e = exp(-im*H1_eff*(T - t0))*M1_vec