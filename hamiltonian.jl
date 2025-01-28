using LinearAlgebra
using ITensors, ITensorMPS
using SparseArrays

include("vectorization.jl")

#Create different operators for use in ITensor opsum operation
ITensors.op(::OpName"Sx2", ::SiteType"Qubit") = [0 1; 1 0]
ITensors.op(::OpName"Sy2", ::SiteType"Qubit") = [0 -im; im 0]
ITensors.op(::OpName"Sz2", ::SiteType"Qubit") = [1 0; 0 -1]
ITensors.op(::OpName"S+2", ::SiteType"Qubit") = [0 2; 0 0]
ITensors.op(::OpName"a+a", ::SiteType"Qubit") = [0 0; 0 1]
ITensors.op(::OpName"aa+", ::SiteType"Qubit") = [1 0; 0 0]
ITensors.op(::OpName"a", ::SiteType"Qubit") = [0 1; 0 0]
ITensors.op(::OpName"a+", ::SiteType"Qubit") = [0 0; 1 0]
ITensors.op(::OpName"-Sy2", ::SiteType"Qubit") = [0 im; -im 0]

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]
splus = sx + im.*sy
sminus = sx - im.*sy
a = [0 0; 0 1]

function matrix_form(MPO::MPO, sites)
    N = length(MPO)
    Matrix_Form = zeros(ComplexF64, (2^N, 2^N))
    for i = 1:2^N
        vec = zeros(2^N)
        vec[i] = 1.0
        vec_mps = MPS(vec, sites)
        mpo_col = MPO*vec_mps
        sites2 = siteinds(mpo_col)
        mpo_col = reconstruct_arr(2, N, mpo_col, sites2)
        Matrix_Form[:,i] = mpo_col
    end
    return Matrix_Form
end


function s_op(op, j, N)
    if j == 1 || j == N + 1
        Ident = Matrix(I, 2^(N - 1), 2^(N - 1))
        return kron(op, Ident)
    elseif j == N
        Ident = Matrix(I, 2^(j - 1), 2^(j - 1))
        return kron(Ident, op)
    else 
        I1 = Matrix(I, 2^(j - 1), 2^(j - 1))
        I2 = Matrix(I, 2^(N - j), 2^(N - j))
        return kron(I1, op, I2)
    end
end

function s_op_sparse(op, j, N)
    op = sparse(op)
    if j == 1 ||j == N + 1
        Ident = sparse(I, 2^(N - 1), 2^(N - 1))
        return kron(op, Ident)
    elseif j == 1
        Ident = sparse(I, 2^(j - 1), 2^(j - 1))
        return kron(Ident, op)
    else
        I1 = Matrix(I, 2^(j - 1), 2^(j - 1))
        I2 = Matrix(I, 2^(N - j), 2^(N - j))
        return kron(I1, op, I2)
    end
end

#All of the below functions are for different Hamiltonian models
#Hamiltonian model without rotational frequency change and with time dependent term
function time_MPO(t, p, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += p(t, N - i + 1), "Sx2", i
        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

#Same as above, exc ept the time dependent part is in terms of a vector instead of a function
function time_MPO_param(step, pt0, qt0, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += pt0[i,step], "Sx2", N - i + 1
        os += qt0[i,step], "-Sy2", N - i + 1
        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

#Simple Ising Model (https://en.wikipedia.org/wiki/Ising_model#Definition) with J_ij = 0, mu = 0 as an MPO
function ising_mpo(N, sites)
    # Make N S=1/2 spin indices
    # sites = siteinds("S=1/2",N)
    # Input the operator terms
    os = OpSum()
    for i=1:N-1
        os += "Sz2",i,"Sz2",i+1
    end
    # Convert these terms to an MPO
    H = MPO(os,sites)
    return H
end

#xxx hamiltonain model as MPO
function xxx_mpo(N, sites, J, g)
    os = OpSum()
    for i = 1:N - 1
        os -= J, "Sz2",i, "Sz2", i + 1
    end
    for i = 1:N
        os -= g*J, "Sx2", i
    end
    H = MPO(os, sites)
    return H 
end

#Ising Model with J_ij = 0, mu = 0 as matrix
function Ising(N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    return H
end

#xxx hamiltonian as a matrix
function xxx(N, J, g)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= -J*s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    for j in 1:N
        H .-= g*J*s_op(sx, j, N)
    end
    return H
end

#Sparse version of xxx hamiltonian matrix
function xxx_sparse(N, J, g)
    H = spzeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H += -J*s_op_sparse(sz, j, N)*s_op_sparse(sz, j + 1, N)
    end
    for j in 1:N 
        H -= g*J*s_op_sparse(sx, j, N)
    end
    return H 
end