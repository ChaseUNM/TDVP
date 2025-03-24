using ITensors, ITensorMPS 

include("hamiltonian.jl")
include("tdvp.jl")
N = 2
d = 2
sites = siteinds("Qudit", N, dim = d)

ground_freq = collect(2:4)
rot_freq = zeros(N)
self_kerr = zeros(N)
cross_kerr = zeros(N, N)
dipole = zeros(N, N)

# H = system_MPO(ground_freq, cross_kerr, dipole, N, sites)

# H = ising_mpo(N, sites)
H = H_MPO_v2(ground_freq, rot_freq, cross_kerr, dipole, N, sites)

println("H1 inds: ", inds(H[1]))
println("H2 inds: ", inds(H[2]))
# println("H3 inds: ", inds(H[3]))
H1 = Array(H[1], inds(H[1]))
H2 = Array(H[2], inds(H[2]))
# H3 = Array(H[3], inds(H[3]))

let 
    for i = 1:dim(inds(H[1])[1])
        println("Bond dimension: $i")
        display(H1[i,:,:])
        display(H2[i,:,:])
        # display(H3[i,:,:])
    end
end

H_mat = matrix_form(H, sites)
display(H_mat)
t0 = 0 
T = 1
steps = 1
init = zeros(ComplexF64, d^N)
init[1] = 1.0 + 0.0*im
init = rand(d^N)
M_init = MPS(init, sites)
println("Linkdims: ", linkdims(M_init))
# M_n, population = tdvp2_constant(H, M_init, t0, T, steps, 0.0)