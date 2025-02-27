using LinearAlgebra, Plots

H_s = [0 0; 0 1]
H_c(t) = [0 sin(t); sin(t) 0]
H(t) = H_s .+ H_c(t)

function sample_function(f::Function, a::Real, b::Real, n::Int)
    x_vals = range(a, b, length=n)  # Generate n evenly spaced points in [a, b]
    return f.(x_vals)  # Apply function f to each point
end

steps = 100
t0 = 0
T = pi
f(x) = 2*sin(10x)
x = sample_function(f, t0, T, steps)
plot(x)

#exponential solver 
init = [1.0 + 0.0*im, 0]
let 
    array = zeros(ComplexF64, (2, steps + 1))
    array[:,1] = init
    h = (T - t0)/steps
    for i in 1:steps
        println("Step $i")
        H = [0 x[i]; x[i] 1]
        u = exp(-im*H*h)*init 
        array[:,i + 1] = u
        init .= u 
    end
    plot(abs2.(array'))
end