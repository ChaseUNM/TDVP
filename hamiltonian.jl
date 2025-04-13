using LinearAlgebra
using ITensors, ITensorMPS
using SparseArrays

include("vectorization.jl")



# Create different operators for use in ITensor opsum operation
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

# a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
# ITensors.op(::OpName"a", ::SiteType"Qudit") = a
# ITensors.op(::OpName"adag", ::SiteType"Qudit") = a'
# ITensors.op(::OpName"a'a'", ::SiteType"Qudit") = a'*a'
# ITensors.op(::OpName"aa", ::SiteType"Qudit") = a*a
# ITensors.op(::OpName"a'a", ::SiteType"Qudit") = a'*a 
# ITensors.op(::OpName"a+a'", ::SiteType"Qudit") = a + a'
# ITensors.op(::OpName"a-a'", ::SiteType"Qudit") = a - a'

function matrix_form(MPO::MPO, sites)
    N = length(MPO)
    d = dim(sites[1])
    Matrix_Form = zeros(ComplexF64, (d^N, d^N))
    for i = 1:d^N
        vec = zeros(d^N)
        vec[i] = 1.0
        vec_mps = MPS(vec, sites)
        mpo_col = MPO*vec_mps
        sites2 = siteinds(mpo_col)
        # mpo_col = reconstruct_arr(d, N, mpo_col, sites2)
        mpo_col = reconstruct_arr_v2(mpo_col)
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

function s_op_general(op, j, N, d)
    if j == 1 || j == N + 1
        Ident = Matrix(I, d^(N - 1), d^(N - 1))
        return kron(op, Ident)
    elseif j == N
        Ident = Matrix(I, d^(j - 1), d^(j - 1))
        return kron(Ident, op)
    else 
        I1 = Matrix(I, d^(j - 1), d^(j - 1))
        I2 = Matrix(I, d^(N - j), d^(N - j))
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
function piecewise_H_MPO(step, pt0, qt0, ground_freq, cross_kerr, dipole, N, sites)
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

function piecewise_H_MPO(step, pt0, qt0, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[N - i + 1] - rot_freq[N - i + 1]
        os += freq, "a'a", i
        os -= 0.5*self_kerr[N - i + 1], "a'a'", i, "aa", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a'a", i, "a'a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
        os += pt0[N - i + 1,step], "a + a'", i
        os += im*qt0[N - i + 1,step], "a - a'", i
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H_MPO_v2(step, pt0, qt0, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[N - i + 1] - rot_freq[N - i + 1]
        os += freq, "adag", i, "a", i
        os -= 0.5*self_kerr[N - i + 1], "adag", i, "adag", i, "a", i, "a", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "adag", i, "a", i, "adag", j, "a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
        os += pt0[N - i + 1,step], "a", i
        os += pt0[N - i + 1,step], "adag", i
        os += im*qt0[N - i + 1,step], "a", i 
        os -= im*qt0[N - i + 1,step], "adag", i
    end
    H = MPO(os, sites)
    return H 
end

function H_MPO_v2(ground_freq, rot_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[i] - rot_freq[i]
        os += freq, "adag", i, "a", i
        os -= 0.5*self_kerr[N - i + 1], "adag", i, "adag", i, "a", i, "a", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "adag", i, "a", i, "adag", j, "a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H_MPO_no_rot(step, pt, qt, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += f[i,step], "Sx2", N - i + 1

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

function piecewise_H(step, f, ground_freq, cross_kerr, dipole, N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for i = 1:N 
        H .+= ground_freq[i]*s_op(a, i, N)
        H .+= pt0[i, step]*s_op([0 1; 1 0], i, N)
        H .+= im*qt0[i, step]*s_op([0 1; -1 0], i, N)
        if i != N 
            for j = i + 1: N 
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op(a, i, N)*s_op(a, j, N)
                #dipole-dipole interaction
                
                H .+= dipole[i,j]*s_op([0 0; 1 0], i, N)*s_op([0 1; 0 0], j, N)
                H .+= dipole[i,j]*s_op([0 1; 0 0], i, N)*s_op([0 0; 1 0], j, N)
            end
        end
    end
    return H 
end

function piecewise_H_no_rot(step, f, ground_freq, cross_kerr, dipole, N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for i = 1:N 
        H .+= ground_freq[i]*s_op(a, i, N)
        H .+= f[i, step]*s_op([0 1; 1 0], i, N)
        if i != N 
            for j = i + 1: N 
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op(a, i, N)*s_op(a, j, N)
                #dipole-dipole interaction
                
                H .+= dipole[i,j]*s_op([0 0; 1 0], i, N)*s_op([0 1; 0 0], j, N)
                H .+= dipole[i,j]*s_op([0 1; 0 0], i, N)*s_op([0 0; 1 0], j, N)
            end
        end
    end
    return H 
end

function H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)
    H = zeros(ComplexF64, (d^N, d^N))
    a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
    for i = 1:N 
        H .+= (ground_freq[i] - rot_freq[i])*s_op_general(a'*a, i, N, d)
        H .-= 0.5*self_kerr[i]*s_op_general(a'*a'*a*a, i, N, d)
        if i != N 
            for j = i + 1: N
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op_general(a'*a, i, N, d)*s_op_general(a'*a, j, N, d)
                #dipole-dipole interaction
                H .+= dipole[i,j]*s_op_general(a', i, N, d)*s_op_general(a, j, N, d)
                H .+= dipole[i,j]*s_op_general(a, i, N, d)*s_op_general(a', j, N, d)
            end
        end
    end
    return H 
end

function H_ctrl(step, p, q, N, d)
    H = zeros(ComplexF64, (d^N, d^N))
    a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
    for i = 1:N 
        H .+= p[i,step]*s_op_general(a + a', i, N, d)
        H .+= im*q[i, step]*s_op_general(a - a', i, N, d)
    end
    return H 
end


function system_MPO(ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
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

function downsample_pulse(pt, qt, nsplines, nsteps)
    if length(pt) == nsteps & length(qt) == nsteps 
        return pt, qt 
    else
        pt_n = zeros(size(pt)[1], nsteps)
        qt_n = zeros(size(qt)[1], nsteps)
        if nsteps % nsplines == 0
            
            for j in 1:size(pt)[1]
                for i in 1:nsplines
                    spline_len = Int64(nsteps/nsplines) 
                    pt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= pt[j, i]
                    qt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= qt[j, i]
                end
            end 
            
        elseif nsteps % nsplines != 0
            println("Number of steps is not divisible by the number of splines") 
            spline_len = Int64(floor(nsteps/nsplines))
            spline_remainder = nsteps % nsplines
            for j in 1:size(pt)[1]
                for i in 1:nsplines
                    if i == nsplines 
                        pt_n[j, (i - 1)*spline_len + 1: i*spline_len + spline_remainder] .= pt[j, i]
                        qt_n[j, (i - 1)*spline_len + 1: i*spline_len + spline_remainder] .= qt[j, i]
                    else
                        pt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= pt[j, i]
                        qt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= qt[j, i]
                    end
                end 
            end
        end 
    end
    return pt_n, qt_n
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

function xxx_v2(N, J, g)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= -g*J*s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    for j in 1:N 
        H .-= J*s_op(sx, j, N)
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