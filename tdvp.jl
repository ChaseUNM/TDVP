using ITensors
using LinearAlgebra


#Helper function that matches indices between an MPO and MPS
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

#Creates effective 2-site Hamiltonian
function effective_Hamiltonian_2site(H, M, i)
    N = length(M)
    H_eff = MPO(N)
    qubit_range = setdiff(collect(1:N), [i, i + 1])
    for j in qubit_range 
        H_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    H_eff[i] = H[i]
    H_eff[i + 1] = H[i + 1]
    return H_eff
end

#Creates effective 0-site Hamiltonian, which is referred to as K so I'm referring to it as the Kamiltonian
function effective_Kamiltonian(H, M)
    N = length(M)
    K_eff = MPO(N)
    for j in 1:N 
        K_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    # println("K_eff: ", K_eff)
    return K_eff 
end

#Performs a single left-to-right sweep of an MPS using the tdvp, evolving forward one time step.
function lr_sweep(H, M, h)
    
    #Ensures orthogonality center is the first site
    orthogonalize!(M, 1)
    # if return_projectors == true
    #     P_L = []
    #     P_R = []
    # end
    N = length(M)
    for i in 1:N - 1 
        # println("Site: ", i)

        #Creates effective Hamiltonian matrices and converts the i-th site to a vector
        H_eff = effective_Hamiltonian(H, M, i)
        H_mat, M_vec = conversion(H_eff, M[i])
        #Evolves M_vec with H_mat with step size 'h'
        M_evolve = exp(-im*H_mat*h)*M_vec

        #Converts back into a tensor
        M_inds = inds(M[i]) 
        M_evolve = ITensor(M_evolve, M_inds)

        #Performs QR decomposition in order to get left-orthogonal tensor
        if i==1
            Q, R = qr(M_evolve, M_inds[1])
        else
            Q, R = qr(M_evolve, M_inds[1:2])
        end

        #Set left-orthogonal tensor as new tensor in MPS
        M[i] = Q

        #Creates effective Kamiltonian matrix and converts the upper triangular part from the QR into a vector
        K_eff = effective_Kamiltonian(H, M)
        K_mat, R_vec = conversion(K_eff, R)
        #Evolves R_vec with K_mat and step size h
        R_evolve = exp(im*K_mat*h)*R_vec

        #Convert R into tensor and multiply it with next tensor in the MPS and then replace
        R_inds = inds(R) 
        R_evolve = ITensor(R_evolve, R_inds)
        M[i + 1] = R_evolve*M[i + 1]
    end
    
    #Performs evolution on last site but without an QR decomposition as the MPS will be completely left-orthogonal.
    H_eff_N = effective_Hamiltonian(H, M, N)
    H_N_mat, M_N_vec = conversion(H_eff_N, M[N])
    M_N_evolve = exp(-im*H_N_mat*h)*M_N_vec 
    M_N_inds = inds(M[N])
    M_N_evolve = ITensor(M_N_evolve, M_N_inds)
    M[N] .= M_N_evolve
    
    #Return completely evolved MPS
    return M 
end

#Performs a single left-to-right sweep of an MPS using the 2 site tdvp, evolving forward one time step.
function lr_sweep_2site(H, M, h, cutoff, verbose)
    
    #Ensures orthogonalityu center is 1
    orthogonalize!(M, 1)
    N = length(M)
    error = 0.0
    for i in 1:N - 1 
        # println("Site $i")
        #Creates the 2-site Hamiltonian matrix and converts the 2 site M block (M[i]*M[i + 1]) to a vector
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        
        M_inds = inds(M_block)

        #Evolves the M block forward with the effective Hamiltonian and convert back into a tensor
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)
        println("(Row, Col): ", size(H_mat_2))
        

        #Performs SVD on the M block to get new left-orthogonal tensor
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))

                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            U, S, V = svd(M_evolve, M_inds[1:2])
            S_diag = diag(Array(S, inds(S)))
            U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1:2], cutoff = cutoff)
            S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
            if verbose == true
                println("Singular Values: ", S_diag)
                println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
            end
            # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
            error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            
        end

        #Set the i-th tensor in MPS to be U which is left-orthogonal
        M[i] = U_trunc
        M_n = S_trunc*V_trunc

        #If we're not on the last M block then evolve the (S*V) tensor with the effective Hamiltonian
        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds)
            #Set next tensor to evolved (S*V) tensor
            M[i + 1] = M2_evolve
        elseif i == N - 1
            #If on last site no evolution takes places
            M[i + 1] = S_trunc*V_trunc
        end
        
    end
    if verbose == true
        println("Error: ", error)
    end
    return M, error
end


function tdvp_constant(H, init, t0, T, steps, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    
    #Run time stepper
    for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        init_copy = lr_sweep(H, init_copy, h)
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp2_constant(H, init, t0, T, steps, cutoff, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create arry to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    truncation_err = zeros(steps + 1)
    truncation_err[1] = 0.0
    link_dim = []
    #Run time stepper
    for i = 1:steps
        println("Step $i")
        init_copy, err = lr_sweep_2site(H, init_copy, h, cutoff, verbose)
        println(length(init_copy))
        truncation_err[i + 1] = err
        if verbose == true
            println("Step: ", i)
            println("Link Dimensions at step $i: ", linkdims(init_copy))
        end
        display(reconstruct_arr_v2(init_copy))
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
        push!(link_dim, prod(linkdims(init_copy)))
    end
    return init_copy, storage_arr, truncation_err, link_dim
end

function tdvp_time(H, init, t0, T, steps, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)

    #Run time stepper
    for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        if length(h_list) > 0
            init_copy = lr_sweep(H(i), init_copy, h_list[i])
        elseif length(h_list) == 0
            init_copy = lr_sweep(H(i), init_copy, h)
        end
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp2_time(H, init, t0, T, steps, cutoff, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    bond_dim_list = []
    push!(bond_dim_list, prod(linkdims(init_copy)))
    #Run time stepper
    for i = 1:steps

        if length(h_list) > 0
            init_copy, _ = lr_sweep_2site(H(i), init_copy, h_list[i], cutoff, verbose)
        elseif length(h_list) == 0
            init_copy, _ = lr_sweep_2site(H(i), init_copy, h, cutoff, verbose)
        end
        if verbose == true
            println("Step: ", i)
            println("Linkdim: ", linkdims(init_copy))
        end
        
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
        # storage_arr[i + 1,:] = zeros(ComplexF64, d)
        push!(bond_dim_list, prod(linkdims(init_copy)))
        
    end
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr, bond_dim_list
end