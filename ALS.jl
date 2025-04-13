using ITensors, ITensorMPS, LinearAlgebra, Random

include("hamiltonian.jl")
#Implement Alternating Least Squares to solve AX = B

function match_index(M, R)
    M_inds = inds(M)
    R_inds = inds(R)
    ind_match = Int64[]
    for i in 1:length(R_inds) 
        for j in 1:length(M_inds)

            if R_inds[i] == M_inds[j]
                # println("It's a match at: ", [i,j])
                push!(ind_match, j)
            end
        end
    end
    ind = collect(1:length(M_inds))
    ind_no_match = setdiff(ind, ind_match)
    return ind_match, ind_no_match 
end

#Converts a tensor to a vector
function tensor_to_vec(T::ITensor)
    T_inds = inds(T)
    T_arr = Array(T, T_inds)
    # println("T_arr")
    # println("----------------------------")
    # display(T_arr)
    T_vec = reshape(T_arr, dim(T))
    return T_vec 
end

#Converts both an effective Hamiltonian and tensor to a Matrix and vector, respectively
#Then checks if the conversion was correct by doing a multiplcation with this matrix and vector and checks against the tensor multiplication
function conversion(H, M)
    H_contract = contract(H)
    #Gets indices of both tensors and determines which indices are matching and not matching
    H_inds = inds(H_contract)
    M_inds = inds(M)
    match_ind, ind_no_match = match_index(H_inds, M_inds)

    #Gets dimensions for row and columns of matrix
    row_H = 1 
    col_H = 1
    # println("H_inds: ", H_inds)
    for i = 1:length(H_inds) 
        if plev(H_inds[i]) == 1
            row_H *= dim(H_inds[i])
        else
            col_H *= dim(H_inds[i])
        end 
    end
    
    #When working the tdvp and tdvp2 there will only be three different situations for the number of the indices
    #below we convert the tensor object into an array in order to convert into a matrix
    if length(H_inds) == 4
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[match_ind[1]], H_inds[match_ind[2]])
    elseif length(H_inds) == 6
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]])
    elseif length(H_inds) == 8
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[ind_no_match[4]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]], H_inds[match_ind[4]])
    end
    #Convert H_arr into a matrix and the M tensor into a vector
    H_mat = reshape(H_arr, (row_H, col_H))
    M_arr = Array(M, M_inds)
    M_vec = reshape(M_arr, dim(M))

    #Calculate both possible multiplcations in order to test if conversion was correct 
    mult1 = H_mat*M_vec
    mult2 = tensor_to_vec(H_contract*M)

    # println("Size of matrix: $(size(H_mat))")

    #Returns H_mat and M_vec if conversion was successful, otherwise doesn't return and gives the error between the multiplcations
    if norm(mult1 - mult2) < 1E-12
        # println("Multiplication difference: ", norm(mult1 - mult2))
        return H_mat, M_vec 
    else 
        println("Tensors not correctly converted to matrix/vector")
        println("Norm error: ", norm(mult1 - mult2))
        # return H_mat, M_vec
    end
end

#Creates effective Hamiltonian
function effective_Hamiltonian(H, M, i)
    N = length(M)
    H_eff = MPO(N)
    qubit_range = setdiff(collect(1:N), i)
    # println("Qubit range: ", qubit_range)
    for j in qubit_range
        H_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    # println(H[i])
    H_eff[i] = H[i]
    # println("Contracted inds: ", inds(contract(H_eff)))
    return H_eff
end

function projectedMPS(M, RHS, i)
    N = length(M)
    M_eff = MPS(N)
    qubit_range = setdiff(collect(1:N), i)
    for j in qubit_range
        M_eff[j] = conj(M[j])*RHS[j]
    end
    M_eff[i] = M[i]
    return M_eff
end

function alternating_ls(H, M_init, RHS)
    N = length(H)
    M_copy = copy(M_init)
    orthogonalize!(M_init, 1)
    for i in 1:N
        H_eff = effective_Hamiltonian(H, M_copy, i)
        RHS_eff = projectedMPS(H, RHS, i)
        println(RHS_eff)
        H_eff_mat, RHS_vec = conversion(H_eff, RHS_eff)
        M_inds = inds(M[i])
        sol = H_eff_mat\RHS_vec
        sol_tensor = ITensor(sol, M_inds)
        if i == 1
            Q, R = qr(sol_tensor, M_inds[1])
        elseif 2 < i < N - 1
            Q, R = qr(sol_tensor, M_inds[1:2])
        elseif i == N
            Q, R = qr(sol_tensor, M_inds[2])
        end
        M_copy[i] = Q
        if i != N 
            M_copy[i + 1] = M_copy[i + 1]*R
        end
    end
    return M_copy 
end


Random.seed!(42)
N = 2
sites = siteinds("Qubit", N)
H = xxx_mpo(N,sites,  1, 1)
H_mat = xxx(N, 1, 1)
init = rand(2^N)
init = init/norm(init)
rhs = ones(2^N) 
rhs = rhs/norm(rhs)

init_mps = MPS(init, sites)
rhs_mps = MPS(rhs, sites)

x = H_mat\rhs 
display(x)

M_sol = alternating_ls(H, init_mps, rhs_mps)

        
