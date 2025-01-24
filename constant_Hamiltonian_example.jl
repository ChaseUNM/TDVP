using ITensors, ITensorMPS
using Plots; pyplot()

include("hamiltonian.jl")
include("tdvp.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 2 
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)

#Set initial condition to be [1 0 0 0]
init = zeros(ComplexF64,2^N)
init[1] = 1.0 + 0.0*im 
M_init = MPS(init, sites)


#Set initial time, final time, and number of steps
t0 = 0.0
T = 10.0
steps = 1000

#Run tdvp 
M_n, population = tdvp(H, M_init, t0, T, steps)

#Reset initial condition
M_init = MPS(init, sites)

#Run tdvp2
M_n_2, population2 = tdvp2(H, M_init, t0, T, steps)

#If true, will test tdvp against matrix exponentian time-stepping
test_tdvp = true

let
    storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
    storage_arr[1,:] = init
    step_size = (T - t0)/steps
    sites = siteinds("Qubit", N)
    for i in 1:steps 
        u = exp(-im.*H_mat.*step_size)*init
        init .= u
        storage_arr[i + 1,:] = init
    end
    #Will plot the errors between both the tdvp methods and matrix exponential time stepping
    if test_tdvp == true
        y1 = plot(range(0, steps).*(T/steps), abs.(abs2.(storage_arr) - abs2.(population)), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 1TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"])
        y2 = plot(range(0, steps).*(T/steps), abs.(abs2.(storage_arr) - abs2.(population2)), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 2TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"], reuse = false)
        plots = plot(y1, y2)
        display(plots)
    end
end
