using ITensors, ITensorMPS

include("vectorization.jl")

x = [1,0,0,1]
x = x/norm(x)

sites = siteinds("Qubit", 2)

M = MPS(x, sites, linkdims = 1)

display(reconstruct_arr_v2(M))
