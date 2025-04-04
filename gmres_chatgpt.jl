using LinearAlgebra

include("hamiltonian.jl")

function gram_schmidt(v, basis, tol)
    orthogonal_v = v
    count = 0
    for b in basis
        count += 1
        if norm(orthogonal_v) < tol
            println("orthogonal_v norm too small")
            break
        end
        # Project v onto b using the conjugate dot product for complex vectors
        projection = (dot(b, v) / dot(b, b)) * b
        orthogonal_v -= projection
    end
    return orthogonal_v/norm(orthogonal_v)
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

function gmres(A, b; tol=1e-6, max_iter=1000)
    n = length(b)
    x = zeros(n)  # Initial guess
    r = b - A * x
    beta = norm(r)

    if beta < tol
        return x
    end

    V = [r / beta]  # Krylov subspace basis
    H = zeros(max_iter + 1, max_iter)
    g = zeros(max_iter + 1)
    g[1] = beta

    for j in 1:max_iter
        # Apply Arnoldi process with your function
        w, norm_w = ApplyOrthonormlize(A, V, tol)
        
        if norm_w < tol
            println("Breakdown in Arnoldi process")
            break
        end

        # Expand Krylov subspace
        push!(V, w)

        # Fill Hessenberg matrix H
        for i in 1:j
            H[i, j] = dot(V[i], A * V[j])  # Projection
        end
        H[j+1, j] = norm_w

        # Solve least squares problem using QR decomposition
        Q, R = qr(H[1:j+1, 1:j])
        y = R \ (Q' * g[1:j+1])

        # Compute solution update
        x += reduce(+, (V[i] * y[i] for i in 1:j))

        # Compute residual
        r = b - A * x
        residual = norm(r)
        
        if residual < tol
            return x
        end
    end

    return x
end

using LinearAlgebra

function gmres(A, b; restart=10, tol=1e-6, max_iter=1000)
    n = length(b)
    x = zeros(n)  # Initial guess
    r = b - A * x
    beta = norm(r)

    if beta < tol
        return x
    end

    for _ in 1:max_iter
        # Restarted GMRES
        V = [r / beta]  # Krylov subspace basis
        H = zeros(restart + 1, restart)
        g = zeros(restart + 1)
        g[1] = beta

        for j in 1:restart
            # Apply Arnoldi process with your function
            w, norm_w = ApplyOrthonormlize(A, V, tol)
            
            if norm_w < tol
                println("Breakdown in Arnoldi process")
                break
            end
            
            # Expand Krylov subspace
            push!(V, w)

            # Fill Hessenberg matrix H
            for i in 1:j
                H[i, j] = dot(V[i], A * V[j])  # Projection
            end
            H[j+1, j] = norm_w

            # Solve least squares problem using QR decomposition
            Q, R = qr(H[1:j+1, 1:j])
            y = R \ (Q' * g[1:j+1])

            # Compute residual norm
            residual = abs(g[j+1])
            if residual < tol
                x += reduce(+, (V[i] * y[i] for i in 1:j))
                return x
            end
        end

        # Update solution
        x += reduce(+, (V[i] * y[i] for i in 1:restart))
        r = b - A * x
        beta = norm(r)

        if beta < tol
            return x
        end
    end
    return x
end

# Example usage
A = xxx(2, 1, 1)
b = ones(4)
b = b/norm(b)
x = gmres(A, b)
println("Solution: ", x)
display(A*x - b)