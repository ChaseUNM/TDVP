using ITensors, ITensorMPS
using Plots

include("hamiltonian.jl")
include("tdvp.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 6
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)

#Set initial condition to be [1 0 0 0]
init = zeros(ComplexF64,2^N)
init[2] = 1.0 + 0.0*im 
M_init = MPS(init, sites)

println("Link Dimensions before: ", linkdims(M_init))
#Set initial time, final time, and number of steps
t0 = 0.0
T = 40.0
steps = 200

#Run tdvp 
M_n, population = tdvp_constant(H, M_init, t0, T, steps)
println("Link Dimensions after TDVP:", linkdims(M_n))
#Reset initial condition
M_init = MPS(init, sites)

#Run tdvp2
M_n_2, population2 = tdvp2_constant(H, M_init, t0, T, steps, 1E-18)
println("Link Dimensions after TDVP2: ", linkdims(M_n_2))
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
        err_tdvp = zeros(size(storage_arr)[1])
        err_tdvp2 = zeros(size(storage_arr)[1])
        for i in 1:size(storage_arr)[1]
            err_tdvp[i] = norm(storage_arr[i,:] - population[i,:])
            err_tdvp2[i] = norm(storage_arr[i,:] - population2[i,:])
        end
        println(size(storage_arr))
        y1 = plot(range(0, steps).*(T/steps), err_tdvp, xlabel = "t", ylabel = "error", 
        plot_title = "Error: 1TDVP and Matrix Exponentiation")
        y2 = plot(range(0, steps).*(T/steps), err_tdvp2, xlabel = "t", ylabel = "error", 
        plot_title = "Error: 2TDVP and Matrix Exponentiation", reuse = false)
        plots = plot(y1, y2)
        display(plots)
    end
end

# let
#     err = []
#     for N in 2:8 
#         println("$N qubits")
#         println("--------------------------------------------------------------------")
#         sites = siteinds("Qubit", N)
#         H = xxx_mpo(N, sites, -1, 1)
#         H_mat = xxx(N, -1, 1)
#         # display(H_mat)
#         init = zeros(ComplexF64,2^N)
#         init[1] = 1.0 + 0.0*im 
#         M_init = MPS(init, sites, maxdim = 2)
#         println("Linkdims before: ", linkdims(M_init))
#         t0 = 0.0
#         T = 1.0
#         steps = 1
#         M_N, population = tdvp2_constant(H, M_init, t0, T, steps)
#         println("Linkdims after: ", linkdims(M_N))
#         u = exp(-im*H_mat)*init
#         println("Error: ", norm(population[end,:] - u))
#         push!(err, norm(population[end,:] - u))
#         println()
#     end
#     p1 = plot(range(2, 8), err, xlabel = "# of qubits", ylabel = "TDVP Error", plot_title = "Max Bond Dimension")
#     display(p1)
#     # savefig(p1, "maxdim1_tdvp2.png")
# end