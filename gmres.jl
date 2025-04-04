using ITensors, ITensorMPS, LinearAlgebra
using Random


include("hamiltonian.jl")
function gram_schmidt(v, basis, tol)
    orthogonal_v = copy(v)
    count = 0
    for b in basis
        count += 1

        # Project v onto b using the conjugate dot product for complex vectors
        projection = (dot(b, orthogonal_v) / dot(b, b)) * b
        orthogonal_v -= projection
        if norm(orthogonal_v) < tol
            println("orthogonal_v norm too small")
            break
        end
    end
    if norm(orthogonal_v) < tol
        println("Norm of krylov vector: ", norm(orthogonal_v))
        return orthogonal_v
    else
        return orthogonal_v/norm(orthogonal_v)
    end
end

function ApplyOrthonormlize(H, krylov_vec, tol)
    # w_k = apply(H, krylov_vec[end])
    w_k = H*krylov_vec[end]
    # println("w_k:", w_k)
    # w = orthonormalize(w_k, krylov_vec, tol)
    w = gram_schmidt(w_k, krylov_vec, tol)
    norm_w = norm(w)
    return w, norm_w
end

function orthoBasis(H, init, tol)
    N = length(init)
    krylov_vec = []
    push!(krylov_vec, init)
    for i = 2:N 
        w, w_norm = ApplyOrthonormlize(H, krylov_vec, tol)
        push!(krylov_vec, w)
    end
    return krylov_vec
end

Random.seed!(43)
init = ones(4)
init = init/norm(init)
krylov_vec = []
push!(krylov_vec, init)
H = rand(4, 4)
w, norm_w = ApplyOrthonormlize(H, krylov_vec, 1E-15)
display(w)
display(norm_w)
push!(krylov_vec, w)
w2, norm_w2 = ApplyOrthonormlize(H, krylov_vec, 1E-15)
display(w2)
push!(krylov_vec, w2)
w3, norm_w3 = ApplyOrthonormlize(H, krylov_vec, 1E-15)
display(w3'*w3)

k_basis = orthoBasis(H, init, 1E-15)
println("Checking: ", k_basis[3]'*k_basis[4])


function updateT(T, H, krylov_vec)
    n = length(krylov_vec)
    T_length = length(T)
    N = Int64(sqrt(T_length))
    T_n = zeros(ComplexF64, (n, n))
    # println("T_prev: ", T)
    T_n[1:N, 1:N] .= T
    # println("Length of T:", N)
    # println("Length of n:", n)
    for i in 1:n
        if i != n
            T_n[i, n] = krylov_vec[i]'*H*krylov_vec[n]
            T_n[n ,i] = krylov_vec[n]'*H*krylov_vec[i]
        else
            T_n[i,i] = krylov_vec[i]'*H*krylov_vec[i]
        end
    end

    # for i = N + 1:n
    #     for j = N + 1:n
    #         # T_n[i,j] = apply(krylov_vec[i], H, krylov_vec[j])
    #         # println("Adjoint stuff: ", adjoint(krylov_vec[i])*H*krylov_vec[j])
    #         T_n[i,j] = adjoint(krylov_vec[i])*H*krylov_vec[j]
    #         println("i,j: ", (i,j))
    #         println(dot(krylov_vec[i], H*krylov_vec[j]))
    #     end
    # end
    return T_n
end

function gram_schmidt_MPS(v, basis, tol)
    orthogonal_v = noprime(v)
    count = 0
    for b in basis
        count += 1
        # ortho = sqrt(abs(real(inner(orthogonal_v, orthogonal_v))))
        ortho = norm(orthogonal_v)
        
        if ortho < tol
            println("Tol reached")
            break
        end
        # Project v onto b using the conjugate dot product for complex vectors
        projection = (inner(b, v) / inner(b, b)) * b
        # orthogonal_v = -(orthogonal_v, projection)
        
        orthogonal_v -= projection
        # println(norm(orthogonal_v))
    end
    # ortho = sqrt(abs(real(inner(orthogonal_v, orthogonal_v))))
    ortho = norm(orthogonal_v)
    if ortho < tol 
        return orthogonal_v, ortho
    else
        return orthogonal_v/ortho, ortho
    end
end

function ApplyOrthonormlize_MPS(H, krylov_vec, tol)

    w_k = H*krylov_vec[end]

    # , mindim = 4, maxdim = 4, cutoff = 0, use_relative_cutoff = false, use_absolute_cutoff = true)
    # println(bond_dimension(w_k))
    # truncate!(w_k, maxdim = 5, cutoff = nothing)
    # w_k = H*krylov_vec[end]
    w, norm_w = gram_schmidt_MPS(w_k, krylov_vec, tol)
    # w = gram_schmidt(w_k, krylov_vec, tol)
    return w, norm_w
end

function updateT_MPS(T, H, krylov_vec)
    n = length(krylov_vec)
    T_length = length(T)
    N = Int64(sqrt(T_length))
    T_n = zeros(ComplexF64, (n, n))
    # println("T_prev: ", T)
    T_n[1:N, 1:N] .= T
    # println("Length of T:", N)
    # println("Length of n:", n)
    T_n[n, n] = inner(krylov_vec[n], H, krylov_vec[n])
    if n > 1
        T_n[n - 1, n] = inner(krylov_vec[n - 1], H, krylov_vec[n])
        T_n[n, n - 1] = inner(krylov_vec[n], H, krylov_vec[n - 1])
    end
    # for i in 1:n
    #     if i != n
    #         T_n[i, i + 1] = inner(krylov_vec[i], H, krylov_vec[n])
    #         T_n[n ,i] = inner(krylov_vec[n], H, krylov_vec[i])
    #     else
    #         T_n[i,i] = inner(krylov_vec[i], H, krylov_vec[i])
    #     end
    # end

    return T_n
end



function formHessenberg(H, krylov_vec)
    n = length(krylov_vec)
    Hess = zeros(ComplexF64, (n, n - 1))
    for j = 1:n - 1
        for i = j:n
            println("(row, col): ($i, $j)") 
            Hess[i,j] = krylov_vec[i]'*H*krylov_vec[j]
        end
    end
    return Hess 
end

function formHessenberg_MPS(H, krylov_vec)
    n = length(krylov_vec)
    Hess = zeros(ComplexF64, (n, n - 1))
    for i = 1:n 
        for j = i  + 1:n - 1
            Hess[i,j] = inner(krylov_vec[i], H, krylov_vec[j])
        end
    end
    return Hess 
end

function gmres_mps(H, init, rhs, tol)
    krylov_vec = []
    N = dim(siteinds(init))
    r0 = rhs - noprime(H*init)
    k0 = r0/norm(r0)
    push!(krylov_vec, r0)
    T = inner(k0, H, k0)
    r0_norm = norm(noprime(H*init) - rhs)
    for i in 2:N
        println("Krylov Vector: $i")
        w, norm_w = ApplyOrthonormlize_MPS(H, krylov_vec, tol)
        push!(krylov_vec, w)
        
        T = formHessenberg_MPS(H, krylov_vec)
        display(T)
        # T0 .= T
        e1 =  zeros(i)
        e1[1] = 1.0
        y = T\(r0_norm*e1)
        r_n = norm(T*y - r0_norm*e1)
        if r_n < tol
            println("Tolerance reached: ", r_n) 
            for j = 1:length(krylov_vec) - 1
                init += y[j]*krylov_vec[j]
            end
            break
        end
        
    end
    return init 
end

function gmres_v1(H, init, rhs, tol)
    krylov_vec = []
    N = length(init)
    r0 = rhs - H*init
    k0 = r0/norm(r0)
    push!(krylov_vec, k0)
    T = adjoint(k0)*H*k0
    r0 = norm((H*init) - rhs)
    println("R0: ", r0)
    println("init: ", init)
    init_copy = copy(init)
    for i in 2:N

        println("Krylov Vector: $i")
        w, norm_w = ApplyOrthonormlize(H, krylov_vec, tol)
        push!(krylov_vec, w)
        println("krylov_vec list: ", w)
        T = formHessenberg(H, krylov_vec)
        println(size(T))
        H_tilde = zeros(ComplexF64, (i + 1, i))
        # H_tilde[1:i, 1:i] .= T 
        # H_tilde[i + 1, i] = krylov_vec[end]'*H*krylov_vec[end - 1]
        # T0 .= T
        e1 =  zeros(i)
        e1[1] = 1.0
        println(e1)
        display(T)
        y = T\(r0*e1)
        println("r0: ", r0)
        display(T)
        println("Y: ", y)
        println("Length: ", length(krylov_vec))
        r_n = norm(T*y - r0*e1)
        println("Residual norm: ", r_n)
        if r_n < 1E-10
            for j = 1:length(krylov_vec) - 1
                init_copy += y[j]*krylov_vec[j]
            end
        break 
        end
    end
    return init_copy, krylov_vec
end

Random.seed!(42)

N = 3 
sites = siteinds("Qubit", N)
H_MPO = xxx_mpo(N, sites, 1, 1)
H_mat = xxx(N, 1, 1)
display(H_mat)

RHS = ones(2^N)
RHS = RHS/norm(RHS)
init = zeros(2^N)
# init = init/norm(init)

RHS_MPS = MPS(RHS, sites)
init_MPS = MPS(init, sites)



x = gmres_mps(H_MPO, init_MPS, RHS_MPS, 1E-15)

x_vec, k_basis = gmres_v1(H_mat, init, RHS, 1E-16)

println(linkdims(x))
display(reconstruct_arr_v2(x))
display(x_vec)
display(H_mat\RHS)
# display(H_mat*x - RHS)
# display(H_mat*x_vec - RHS)
# # display(reconstruct_arr_v2(x))
