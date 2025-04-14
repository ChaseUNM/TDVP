using ITensors, ITensorMPS, LinearAlgebra 

include("hamiltonian.jl")
include("tdvp.jl")

#Create Manual MPO and use that for a manual 2 site TDVP
N = 2
sites = siteinds("Qubit", N)

H1 = zeros(3, 2, 2)
H1[1,:,:] = [1 0; 0 -1]
H1[2,:,:] = [0 1; 1 0]
H1[3,:,:] = [-1 0; 0 -1]
H2 = zeros(3, 2, 2)
H2[1,:,:] = [1 0; 0 -1]
H2[2,:,:] = [-1 0; 0 -1]
H2[3,:,:] = [0 1; 1 0]

b1 = Index(3, "link, l = 1")

H1_tens = ITensor(H1, b1, sites[1], sites[1]')
H2_tens = ITensor(H2, b1, sites[2], sites[2]')

H_MPO = MPO(N)
H_MPO[1] = H1_tens
H_MPO[2] = H2_tens

H_mat = xxx_v2(N, 1, -1)
println("Norm of difference between MPO and Mat: ", norm(H_mat - matrix_form(H_MPO, sites)))

init = [1, 0, 0, 1]
init = init/norm(init)

M_init = MPS(init, sites)
orthogonalize!(M_init, 1)

H_eff_1 = effective_Hamiltonian(H_MPO, M_init, 1)
H_eff_1_mat, M1_vec = conversion(H_eff_1, M_init[1])
println("Manual H_mat")
display(H_eff_1_mat)
display(M1_vec)
M1_vec = exp(-im*H_eff_1_mat)*M1_vec
println("M1_vec evolve")
display(M1_vec)
M1 = ITensor(M1_vec, inds(M_init[1]))
Q, R = qr(M1, inds(M_init[1])[1])

M_init[1] = Q

K_eff_1 = effective_Kamiltonian(H_MPO, M_init)
K_eff_1_mat, C1_vec = conversion(K_eff_1, R)
println("Manual K_mat")
display(K_eff_1_mat)
display(C1_vec)
C1_vec = exp(im*K_eff_1_mat)*C1_vec 
println("Manual R_evolve")
display(C1_vec)
C1 = ITensor(C1_vec, inds(R))
M_init[2] = C1*M_init[2]
println("This is what you want!")
println(M_init[2])


H_eff_2 = effective_Hamiltonian(H_MPO, M_init, 2)
H_eff_2_mat, M2_vec = conversion(H_eff_2, M_init[2])
println("Manual H_mat 2")
display(H_eff_2_mat)
println("Manual M2_vec pre-evolve")
display(M2_vec)
M2_vec = exp(-im*H_eff_2_mat)*M2_vec
println("Manual M2_vec evolve")
display(M2_vec) 
M2 = ITensor(M2_vec, inds(M_init[2]))
M_init[2] .= M2

init_evolve = exp(-im*H_mat)*init

display(reconstruct_arr_v2(M_init))
display(init_evolve)
M_init_2 = MPS(init, sites)

println("*******************************************")
M_evolve, _ = tdvp_constant(H_MPO, M_init_2, 0.0, 1.0, 1)
display(reconstruct_arr_v2(M_evolve))

println("Difference between evolution: ", norm(reconstruct_arr_v2(M_evolve) - reconstruct_arr_v2(M_init)))
#Create manual MPO and use that for a manual 3 site TDVP 
#=
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
=#