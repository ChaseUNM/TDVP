using ITensors, ITensorMPS 
using LinearAlgebra, Random

include("tdvp.jl")

Random.seed!(42)
N = 4 
x = rand(2^N)
x = x/norm(x)

t0 = 0
T = 1
delta_t = (T - t0)

sites = siteinds("Qubit", N)
M = MPS(x, sites)

H_MPO = xxx_mpo(N, sites, 1, 1)

orthogonalize!(M, 1)

#Construct first projector, I_2 \otimes P_R^{[2:3]}
B2t3 = contract(M[2]*M[3])

B2t3p = prime(B2t3, inds(B2t3)[1], inds(B2t3)[3])

PR2t3 = B2t3*conj(B2t3p)
PR2t3_arr = Array(PR2t3, inds(PR2t3))
PR2t3_mat = reshape(PR2t3_arr, (4,4))
display(PR2t3_mat)
function prime_sites(T)
    T_ind = inds(T)
    T_copy = T
    for i in T_ind 
        if hastags(i, "Qubit") == true 
            T_copy = prime(T_copy, i)
        end
    end
    return T_copy
end

function projector(M, start_site, end_site)
    M_prod = M[start_site]
    for i in collect(start_site + 1: end_site)
        M_prod *= M[i]
    end
    M_prod_p = prime_sites(M_prod)
    P = M_prod*conj(M_prod_p)
    P_inds = inds(P)
    row = 1
    col = 1
    for i in P_inds
        if plev(i) == 0
            row *= dim(i)
        elseif plev(i) == 1
            col *= dim(i)
        end
    end
    P_arr = Array(P, P_inds)
    P_mat = reshape(P_arr, (row, col))
    return P_mat
end

function lr_sweep_test(H, M, h)
    N = length(M)
    orthogonalize!(M, 1)
    P_L = []
    P_R = []
    push!(P_L, Matrix(1.0*I, 2, 2))
    push!(P_R, projector(M, 2, N))
    #Ensures orthogonality center is the first site
    
    # if return_projectors == true
    #     P_L = []
    #     P_R = []
    # end
    
    for i in 1:N - 1 
        println("Site: ", i)

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
        push!(P_L, projector(M, 1, i))
        push!(P_R, projector(M, i + 1, N))

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
    return M, P_L, P_R
end

# _, P_L, P_R = lr_sweep_test(H_MPO, M, delta_t)

# let
#     for i in 1:length(P_L)
#         display(kron(P_L[i], P_R[i]))
#     end
# end
