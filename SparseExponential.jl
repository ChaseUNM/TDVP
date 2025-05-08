using LinearAlgebra, SparseArrays, ExponentialUtilities

include("hamiltonian.jl")


let
    N = 3
    d = 2

    init = zeros(d^N)
    init[1] = 1.0
    init = sparse(init)
    H = xxx(N, 1, 1)
    H_sparse = xxx_sparse(N, 1, 1)

    h = 0.01
    steps = 10
    for i = 1:steps
        init = init + h*H_sparse*init
        display(init)
    end
end