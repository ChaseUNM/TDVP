using ITensors
using LinearAlgebra

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

function tensor_to_vec(T::ITensor)
    T_inds = inds(T)
    T_arr = Array(T, T_inds)
    # println("T_arr")
    # println("----------------------------")
    # display(T_arr)
    T_vec = reshape(T_arr, dim(T))
    return T_vec 
end

function conversion(H, M)
    H_contract = contract(H)
    # println("M: ")
    # println("--------------------------------")
    # println(M_contract)
    # println("R: ")
    # println("--------------------------------")
    # println(R)
    H_inds = inds(H_contract)
    # println(M_inds)
    # println(M)
    M_inds = inds(M)
    # println(R_inds)
    # println("H_inds: ", H_inds)
    # println("M_inds: ", M_inds)
    match_ind, ind_no_match = match_index(H_inds, M_inds)
    # println(match_ind)
    # println(ind_no_match)
    # println("Number of open indices in H_eff: ", length(H_inds))
    row_H = 1 
    col_H = 1
    for i = 1:length(H_inds) 
        if plev(H_inds[i]) == 1
            row_H *= dim(H_inds[i])
        else
            col_H *= dim(H_inds[i])
        end 
    end
    # println(length(H_inds))
    if length(H_inds) == 4
        # M_arr = Array(M_contract, M_inds[match_ind[1]], M_inds[match_ind[2]], M_inds[ind_no_match[1]], M_inds[ind_no_match[2]])
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[match_ind[1]], H_inds[match_ind[2]])
    elseif length(H_inds) == 6
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]])
    elseif length(H_inds) == 8
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[ind_no_match[4]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]], H_inds[match_ind[4]])
    end
    # println("M_arr")
    # println(M_contract)
    # H_arr2 = Array(H_contract, (H_inds[1], H_inds[2], H_inds[3], H_inds[4]))
    # display(M_arr2[:,1,:,2])
    # display(M_arr)
    H_mat = reshape(H_arr, (row_H, col_H))
    # display(M_mat)
    M_arr = Array(M, M_inds)
    # println("R_arr")
    # display(R_arr)
    M_vec = reshape(M_arr, dim(M))
    # display(R_vec)
    # M_arr = Array(M_contract, (M_inds[1], M_inds[2], M_inds[3], M_inds[4]))
    # println("R vec: ")
    # display(R_vec)
    # println("Multiplication 1: ", M_mat*R_vec)
    # println("Multiplcation 2: ", tensor_to_vec(M_contract*R))
    mult1 = H_mat*M_vec
    mult2 = tensor_to_vec(H_contract*M)
    # println("Multiplication difference: ", norm(mult1 - mult2))
    if norm(mult1 - mult2) < 1E-15
        # println("Multiplication difference: ", norm(mult1 - mult2))
        return H_mat, M_vec 
    else 
        println("Tensors not correctly converted to matrix/vector")
        println("Norm error: ", norm(mult1 - mult2))
        return H_mat, M_vec
    end
end

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

function effective_Kamiltonian(H, M)
    N = length(M)
    K_eff = MPO(N)
    for j in 1:N 
        K_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    # println("K_eff: ", K_eff)
    return K_eff 
end

function lr_sweep(H, M, h)
    # println(orthoCenter(M))
    orthogonalize!(M, 1)
    println(orthoCenter(M))
    N = length(M)
    for i in 1:N - 1 
        println("Site: ", i)
        H_eff = effective_Hamiltonian(H, M, i)
        # println("H_eff: ", H_eff)
        # println("M[1]: ", M[1])
        # println("M[2] : ", M[2])
        H_mat, M_vec = conversion(H_eff, M[i])
        M_evolve = exp(-im*H_mat*h)*M_vec
        M_inds = inds(M[i]) 
        M_evolve = ITensor(M_evolve, M_inds)
        if i==1
            Q, R = qr(M_evolve, M_inds[1])
        else
            Q, R = qr(M_evolve, M_inds[1:2])
        end
        M[i] = Q
        # println("Q: ", Q)
        # println("-------------------------------------")
        # println("M: ", M)
        K_eff = effective_Kamiltonian(H, M)
        K_mat, R_vec = conversion(K_eff, R)
        R_evolve = exp(im*K_mat*h)*R_vec
        R_inds = inds(R) 
        R_evolve = ITensor(R_evolve, R_inds)
        M[i + 1] = R_evolve*M[i + 1]
        # println("M[$i] inds: ",inds(M[i]))
    end
    H_eff_N = effective_Hamiltonian(H, M, N)
    H_N_mat, M_N_vec = conversion(H_eff_N, M[N])
    M_N_evolve = exp(-im*H_N_mat*h)*M_N_vec 
    M_N_inds = inds(M[N])
    M_N_evolve = ITensor(M_N_evolve, M_N_inds)
    M[N] .= M_N_evolve
    
    return M 
end

function lr_sweep_2site(H, M, h)
    orthogonalize!(M, 1)
    N = length(M)
    for i in 1:N - 1 
        println("Site $i")
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        M_inds = inds(M_block)
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)

        if i == 1
            U, S, V = svd(M_evolve, M_inds[1])
        else
            U, S, V = svd(M_evolve, M_inds[1:2])
        end
        M[i] = U
        # println("Inds: ", inds(U))
        M_n = S*V

        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds) 
            M[i + 1] = M2_evolve
        elseif i == N - 1
            M[i + 1] = S*V
        end
        
    end
    return M
end

function tdvp(H, init, t0, T, steps)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    h = (T - t0)/steps
    storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
    storage_arr[1,:] = reconstruct_arr(2, N, init, sites)
    
    fill_int = Int64(steps/2)
    pt = [vcat(fill(0,fill_int), fill(0, fill_int)), vcat(fill(0, fill_int), fill(0, fill_int)), vcat(fill(0, fill_int), fill(0, fill_int)),
    vcat(fill(4, fill_int), fill(4.5, fill_int))]
    
    ground_freq = [4.80595*(2*pi), 4.8601*(2*pi)]
    cross_kerr = [0 0 0 0; 0 0 0 0; 0 0 0 0]
    dipole = [0 .005*(2*pi) 0 0; 0 0 0 0; 0 0 0 0]
    H = xxx_mpo(N, sites, 1, 1)
    for i = 1:steps
        # println("------------------------------------")
        # H = xxx_mpo(N, sites, 1, 1)
        println("Step: ", i)
        # H = time_MPO_param(i, pt, ground_freq, cross_kerr, dipole, N, sites)
        # println("Checking...: ", abs2.(reconstruct_arr(2, N, init, sites)))
        init = lr_sweep(H, init, h)
        # init .= M
        # println("Checking... 2: ", abs2.(reconstruct_arr(2, N, init, sites))) 
        storage_arr[i + 1,:] = reconstruct_arr(2, N, init, sites)

    end
    return init, storage_arr
end

function tdvp2(H, init, t0, T, steps)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    h = (T - t0)/steps
    storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
    storage_arr[1,:] = reconstruct_arr(2, N, init, sites)

    fill_int = Int64(steps/2)
    pt = [vcat(fill(1,fill_int), fill(1.5, fill_int)), vcat(fill(2, fill_int), fill(2.5, fill_int)), vcat(fill(3, fill_int), fill(3.5, fill_int))]
    # vcat(fill(4, fill_int), fill(4.5, fill_int))]
    
    ground_freq = [2 1 3]
    cross_kerr = [0 0 0; 0 0 0; 0 0 0]
    dipole = [0 1 0; 0 0 0; 0 0 0]

    for i = 1:steps
        # println("------------------------------------")
        # H = xxx_mpo(N, sites, 1, 1)
        println("Step: ", i)
        # H = time_MPO_param(i, pt, ground_freq, cross_kerr, dipole, N, sites)
        # H_mat = matrix_form(time_MPO_param(i, pt, [2, 1, 3], [0 0 0; 0 0 0; 0 0 0], [0 1.0 0; 0 0 0; 0 0 0], N, sites), sites)
        # println("Checking...: ", abs2.(reconstruct_arr(2, N, init, sites)))
        println("init: ", init)
        init = lr_sweep_2site(H, init, h)
        println("init: ", init)
        # init .= M
        # println("Checking... 2: ", abs2.(reconstruct_arr(2, N, init, sites))) 
        storage_arr[i + 1,:] = reconstruct_arr(2, N, init, sites)

    end
    return init, storage_arr
end