using ITensors, ITensorMPS 
using LinearAlgebra, Plots

include("hamiltonian.jl")

function sample_function(f::Function, a::Real, b::Real, n::Int)
    x_vals = range(a, b, length=n)  # Generate n evenly spaced points in [a, b]
    return f.(x_vals)  # Apply function f to each point
end

function schmidt_coeff(state)
    mat = reshape(state, 2, 2)'
    U, S, V = svd(mat)
    return S
end

t0 = 0
T = 1400
steps = 50000
h = (T - t0)/steps

f(x) = sin(10*x)
g(x) = sin(12*x)
pulse_1 = sample_function(f, t0, T, steps)
pulse_2 = sample_function(g, t0, T, steps)
pulse = hcat(pulse_1, pulse_2)'
cross_kerr = [0 .005; 0 0]
ground_freq = [4.8, 5.0]
dipole = zeros(2, 2)




s_coeff = schmidt_coeff(init)
display(s_coeff)
H(t) = piecewise_H_no_rot(t, pulse, ground_freq, cross_kerr, dipole, N)
display(H(2))
let 
    N = 2
    population = zeros(ComplexF64, (2^N, steps + 1))
    schmidt_vec = zeros(steps + 1)
    
    init = zeros(ComplexF64, 2^N)
    init[1] = 1.0 + 0.0*im 
    s = schmidt_coeff(init)
    schmidt_vec[1] = minimum(s)
    population[:,1] = init
    for i = 1:steps 
        println("Step $i")
        s = schmidt_coeff(init)
        if minimum(s) < 1E-15
            println("Very small schmidt coefficient")
        else
            println("Smallest Schmidt Coefficient is ", minimum(s))
        end
        schmidt_vec[i + 1] = minimum(s)
        init = exp(-im*H(i)*h)*init
        population[:,i + 1] = init 
    end
    entanglement = plot(range(0,steps)*(T-t0)/steps, schmidt_vec)
    population = plot(range(0, steps)*(T - t0)/steps, abs2.(population'))
    display(entanglement)
end







