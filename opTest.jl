using ITensorMPS, ITensors
using LinearAlgebra

include("hamiltonian.jl")

function ITensors.op(::OpName"P1", ::SiteType"Qudit", d::Int)
    Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
    return o
end

function ITensors.op(::OpName"P1", ::SiteType"Boson")
    o = zeros(3, 3)
    o[1, 1] = 1
    return o
end

d = 3
N = 2
a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))

ITensors.op(::OpName"a", ::SiteType"Qudit") = [0 1 0; 0 0 sqrt(2); 0 0 0]

# ITensors.op(::OpName"a", ::SiteType"Qudit") = [
#     0 1 0;
#     0 0 sqrt(3);
#     0 0 0
# ]


# Example usage
i = Index(3, "Qudit")  # Define a site index with dimension 3
A = op("a", i)         # Get the operator matrix
println(A)


let
    N = 2
    
    os = OpSum()
    for n in 1:N
        os += "a", n, "adag", n, "a", n, "adag", n # Adding the "a" operator to each site
    end
    sites = siteinds("Qudit", N, dim = 3)
    ampo = MPO(os, sites)
    @time begin
    
    M = matrix_form(ampo, sites)
    end
    a = [0 1 0; 0 0 sqrt(2); 0 0 0]
    M_mat = kron(a*a'*a*a', Matrix(1.0*I, 3, 3)) + kron(Matrix(1.0*I, 3, 3), a*a'*a*a')
    display(M_mat)
    display(M)
    println(norm(M - M_mat))
end

# let 
#     N = 2

#     # Create site indices
#     sites = [Index(3, "Qudit") for _ in 1:N]

#     # Initialize an MPO
#     M = MPO(sites)

#     # Fill the MPO with local operators
#     for n in 1:N
#         M[n] = ITensor(sites[n]', sites[n])  # Create an ITensor for the site
#         M[n] .= op("a", sites[n])  # Assign the operator matrix
#     end

#     M_mat = matrix_form(M, sites)
#     # Display the MPO
#     display(M_mat)

# end