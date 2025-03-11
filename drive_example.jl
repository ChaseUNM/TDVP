using LinearAlgebra, Plots

H_s_rot = [0 0; 0 0]
H_c(t) = [0 sin(t); sin(t) 0]
H(t) = H_s .+ H_c(t)

function sample_function(f::Function, a::Real, b::Real, n::Int)
    x_vals = range(a, b, length=n)  # Generate n evenly spaced points in [a, b]
    return f.(x_vals)  # Apply function f to each point
end

steps = 1000
t0 = 0
T = 1
f(x) = sin(x)
f(x) = pi*(2/sqrt(2) + im*(2/sqrt(2)))
x = sample_function(f, t0, T, steps)


#exponential solver 
init = [0.0 + 0.0*im, 1.0 + 0.0*im]
let 
    array = zeros(ComplexF64, (2, steps + 1))
    array[:,1] = init
    h = (T - t0)/steps
    for i in 1:steps
        println("Step $i")
        H = [0 pi*(sqrt(2)/2 + im*(sqrt(2)/2)); pi*(sqrt(2)/2 - im*(sqrt(2)/2)) 0]
        u = exp(-im*H*h)*init 
        array[:,i + 1] = u
        init .= u 
    end
    plot(range(0, steps).*(T - t0)/steps, abs2.(array'))
end