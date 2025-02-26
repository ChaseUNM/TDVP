using ITensors, ITensorMPS
using Plots, LaTeXStrings, LinearAlgebra, Random

Random.seed!(42)
include("hamiltonian.jl")
include("tdvp.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 7
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)

eig = eigvals(H_mat)
F = eigen(H_mat)

U = F.vectors
eigs = F.values
# display(U[:,1])


max_eig = maximum(abs.(eig))
min_eig = minimum(abs.(eig))
println(length(U[1,:]))
println(max_eig)
println(min_eig)
long_period = 2*pi/min_eig 
short_period = 2*pi/max_eig

# freq_01 = [4.80595, 4.8601, 5.0]
# freq_01 .*= 2*pi 

# cross_kerr = [0 0 0; 0 0 0; 0 0 0]
# Jkl = [0 0.005 0; 0 0 0; 0 0 0]
# Jkl .*= 2*pi 

# H = system_MPO(freq_01, cross_kerr, Jkl, N, sites)
# H_mat = matrix_form(H, sites)

#Set initial condition to be [1 0 0 0]
init = zeros(ComplexF64,2^N)
init[1] = 1.0 + 0.0*im
# init[4] = 1.0 + 0.0*im


M_init = MPS(init, sites)

A2 = M_init[2]
A2_arr = Array(A2, inds(A2))
display(A2_arr[:,2,:]*A2_arr[:,2,:]')

# function right_projector

println("Link Dimensions before: ", linkdims(M_init))
#Set initial time, final time, and number of steps
t0 = 0.0
T = 1E-4
steps = 1
step_size = (T - t0)/steps

#Run tdvp 
M_n, population = tdvp_constant(H, M_init, t0, T, steps)

#Compare evolved TDVP with just matrix exponential 
u = exp(-im*H_mat*(T - t0))*init 
println(norm(population[end,:] - u))

# println("Link Dimensions after TDVP:", linkdims(M_n))
#Reset initial condition
M_init = MPS(init, sites)

#Run tdvp2
# M_n_2, population2, truncation_err = tdvp2_constant(H, M_init, t0, T, steps, 1E-8)
# println("Link Dimensions after TDVP2: ", linkdims(M_n_2))
#If true, will test tdvp against matrix exponentian time-stepping
test_tdvp = false
let
    if test_tdvp == true
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

        err_tdvp = zeros(size(storage_arr)[1])
        # err_tdvp2 = zeros(size(storage_arr)[1])
        for i in 1:size(storage_arr)[1]
            err_tdvp[i] = norm(storage_arr[i,:] - population[i,:])
            # err_tdvp2[i] = norm(storage_arr[i,:] - population2[i,:])
        end
        x_range = range(0, steps).*(T/steps)
        pop_plot = plot(x_range, [real(storage_arr[:,1]) imag(storage_arr[:,1])], labels = ["Real Part" "Imaginary Part"], xlabel = "t", dpi = 150)
        savefig("Dynamics1.png")
        # savefig("Dynamics.png")
        #[1/sqrt(2)*cos.(3.49 .*x_range).-1/(2*sqrt(2))*cos.(x_range) 0.29/sqrt(2)*sin.(3.49 .*x_range)+1/(2*sqrt(2))sin.(x_range)]
        # check_plot = plot(x_range, 
        # [0.29/sqrt(2)*cos.(3.49 .*x_range).-1/(2*sqrt(2))*cos.(x_range) real(storage_arr[:,2]) imag(storage_arr[:,2]) 0.29/sqrt(2)*sin.(3.49 .*x_range).+1/(2*sqrt(2))*sin.(x_range)])
        # display(check_plot)
        # y1 = plot(range(0, steps).*(T/steps), err_tdvp, xlabel = "t", ylabel = "error", 
        # plot_title = "Error: 1TDVP and Matrix Exponentiation")
        # display(y1)
        # y2 = plot(range(0, steps).*(T/steps), [err_tdvp2 truncation_err], labels = ["TDVP2 error" "Truncation Error"], xlabel = "t", ylabel = "error", 
        # plot_title = "Error: 2TDVP and Matrix Exponentiation", reuse = false)
        # y3 = plot(range(0, steps), abs2.(population2))
        # plots = plot(y1, y2)
        # savefig(y2, "TruncationErr.png")
        # display(y2)
    end
end



let 
    pts = 20
    max_dim = 8
    err_arr = zeros(max_dim, pts)
    # h = LinRange(1E-, 1E-0, pts)
    h = LinRange(-5, 3, pts)
    h = 10 .^-h 
    h = reverse(h)
    println(h)
    init = zeros(ComplexF64,2^N)
    # init[1] = 1.0 + 0.0*im
    init = U[:,1]
    c = U\init 
    C_diag = diagm(c)
    CU = U*C_diag
    
    
    for max in 1:max_dim
        
        # h_size = (50:-1:5)
        err = zeros(length(h))
        arr_number = 1
        t0 = 0.0
        
        
        
        # init[2] = 1.0/sqrt(2) + 0.0*im
        
        # init = U[1,:]

        #Construct the exact solution using eigenvectors and eigenvalues 
        
        for i in h
                
            M_init = MPS(init, sites, maxdim = max)
            println("Step size #$arr_number")
            # h = 1/i
            T = i
            # steps = Int64((T - t0)*i)
            
            M, population = tdvp_constant(H, M_init, t0, T, 1)
            # println("Link dims: ", linkdims(M))
            u = exp(-im.*H_mat*T)*init
            # expEig = exp.(-im*eigs*T)
            # exactSol = CU*expEig
            err[arr_number] = norm(population[end,:] - u)
            arr_number += 1
        end
        
        err_arr[max, :] = err
        # h = 1 ./h_size
        h2 = h.^2
    end

    p1 = plot(h, err_arr', xscale =:log10, yscale=:log10, labels = ["Max dim = 1" "Max dim = 2" "Max dim = 3" "Max dim = 4" "Max dim = 5" "Max dim = 6" "Max dim = 7" "Max dim = 8"], 
    legend=:topleft, ylabel = "error", xlabel = L"\Delta t", dpi = 150, legendfontsize = 5, xticks = [1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1E0], 
    yticks = [1E-30, 1E-25, 1E-20, 1E-15, 1E-10, 1E-5])
    plot!(h, h.^2, linestyle =:dash, linewidth =:2, linecolor =:black, label = L"(\Delta t)^2")
    display(p1)
    # savefig("ErrorAllTo1.png")
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
#         M_init = MPS(init, sites)
#         println("Linkdims before: ", linkdims(M_init))
#         t0 = 0.0
#         T = 10000.0
#         h = (T - t0)
#         steps = 1
#         M_N, population = tdvp_constant(H, M_init, t0, T, steps)
#         println("Linkdims after: ", linkdims(M_N))
#         u = exp(-im*H_mat*h)*init
#         println("Error: ", norm(population[end,:] - u))
#         push!(err, norm(population[end,:] - u))
#         println()
#     end
#     p1 = plot(range(2, 8), err, xlabel = "# of qubits", ylabel = "TDVP Error", plot_title = "Max Bond Dimension")
#     display(p1)
#     # savefig(p1, "maxdim1_tdvp2.png")
# end