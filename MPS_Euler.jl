using ITensors, ITensorMPS, LinearAlgebra

include("hamiltonian.jl")

function euler_mpo(H, init, t0, T, steps)
    h = (T - t0)/steps 
    init_copy = copy(init)
    vec_state_history = []
    push!(vec_state_history, init_copy)
    for i = 1:steps
        HM = -i*h*H*init_copy
        HM = noprime(HM)
        init_copy = init_copy + HM
        # println("Linkdims: ", linkdims(init_copy))
        truncate!(init_copy, cutoff = 1E-10) 
        push!(vec_state_history, init_copy)
    end
    return init_copy, vec_state_history
end

function euler(H, init, t0, T, steps)
    h = (T - t0)/steps 
    init_copy = copy(init)
    vec_state_history = []
    push!(vec_state_history, init_copy)
    for i = 1:steps 
        init_copy = init_copy - i*h*H*init_copy
        push!(vec_state_history, init_copy)
    end
    return init_copy, vec_state_history
end

N = 2
d = 2
sites = siteinds("Qubit", N)

H = ising_mpo(N, sites)/10
H_mat = Ising(N)/10
init = zeros(ComplexF64, d^N)
init[1] = 1.0/sqrt(2) + 0.0*im
init[2] = 1.0/sqrt(2) + 0.0*im
M_init = MPS(init, sites)
println(siteinds(M_init))
println(siteinds(H*M_init))
println(H_app)
t0 = 0.0
T = 1.0
steps = 100


init_evolve = exp(-im*H_mat*(T - t0))*init
display((init_evolve))

M_evolve, _ = euler_mpo(H, M_init, t0, T, steps)
display(abs2.(reconstruct_arr_v2(M_evolve)))
init_evolve, _ = euler(H_mat, init, t0, T, steps)
display(abs2.(init_evolve))
