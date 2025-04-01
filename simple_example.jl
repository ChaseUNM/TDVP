using LinearAlgebra, Random
using ITensors, ITensorMPS 
include("tdvp(3-24).jl")
include("hamiltonian.jl")

Random.seed!(42)

#Create very simple example with constant example for TDVP2 testing 

let 
    N = 5
    sites = siteinds("Qubit", N)


    H_MPO = xxx_mpo(N, sites, 1, 1)
    
    H_mat = xxx(N, 1,1)
    init = zeros(ComplexF64, 2^N)
    init[1] = 1.0 + 0.0*im
    M_init = MPS(init, sites)
    display(reconstruct_arr_v2(M_init))
    t0 = 0.0
    T = 1.0
    steps = 10000
    h = (T - t0)/steps
    storage_arr = zeros(ComplexF64, (steps + 1, 2^N))
    storage_arr[1,:] = init
    for i in 1:steps 
        init = exp(-im*H_mat*h)*init 
        storage_arr[i + 1,:] = init 
    end 

    M_ev, population = tdvp2_constant(H_MPO, M_init, t0, T, steps, 1E-10)
    p1 = plot(abs2.(population[:,1]))
    display(p1)
    println("Press Enter")
    readline()
    
    p2 = plot(abs2.(storage_arr[:,1]))
    display(p2)
    println("Press Enter")
    readline()
    p3 = plot(abs.(abs2.(storage_arr[:,1]) - abs2.(population[:,1])))
    display(p3)
end

