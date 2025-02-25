using ITensors, ITensorMPS
using Plots, LaTeXStrings, LinearAlgebra

include("hamiltonian.jl")
include("tdvp.jl")

#Create 2 qubit site with the xxx heisenberg model
N = 2
sites = siteinds("Qubit", N)
H = xxx_mpo(N, sites, -1, 1)
H_mat = xxx(N, -1,1)
# display(H_mat)
eig = eigvals(H_mat)
F = eigen(H_mat)
# println(eig)
U = F.vectors
eigs = F.values
# display(U[:,1])
# display(U[:,4])
max_eig = maximum(abs.(eig))
min_eig = minimum(abs.(eig))
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
# init[1] = 1.0 + 0.0*im
# init[4] = 1.0 + 0.0*im
# init = rand(ComplexF64, 2^N)
init = [1, 2, 3, 4]
init = init/norm(init)
# init = U[:,4]
# init = (1/sqrt(2))*F.vectors[:,1] + (1/sqrt(2))*F.vectors[:,6]
M_init = MPS(init, sites, maxdim = 1)
orthogonalize!(M_init, 1)
# println(M_init[1])
# println(M_init[2])
# println(orthoCenter(M_init))
println("-------------B2----------------")
# println(M_init[2])
B2 = Array(M_init[2], inds(M_init[2]))
# println(B2[:,1]'*B2[:,1] + B2[:,2]'*B2[:,2])
# println(B2[:,1]'*B2[:,2])
# println("Right projector: ")
# display(B2*B2')
right_proj = B2*B2'
# display(right_proj) 
# display(right_proj*right_proj)
P1 = kron([1 0; 0 1], right_proj)
first = P1*H_mat*(-im)
first_commute = -im*H_mat*P1 
display(first)
display(first_commute)
display(first - first_commute)
# display((P1*H_mat) - (P1*H_mat)')
println("----------------------------------------------------")

# function right_projector

println("Link Dimensions before: ", linkdims(M_init))
#Set initial time, final time, and number of steps
t0 = 0.0
T = 20
steps = 1
step_size = (T - t0)/steps

#Run tdvp 
M_n, population = tdvp_constant(H, M_init, t0, T, steps)
# println("Pop!!!!: ", population[end,:])
u = exp(-im.*H_mat.*step_size)*init
# println("Pop!!!: ", u)
A1 = Array(M_n[1], inds(M_n[1]))
# println(M_n[1])
# display(A1)
# println(A1[1,:]*A1[1,:]')
# println("Left Projector: ")
left_proj = A1*A1'
# display(left_proj)
# display(left_proj*left_proj)

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
        println(size(population))
        println(size(population[:,4]))
        println(init)
        x_range = range(0, steps).*(T/steps)
        pop_plot = plot(x_range, [real(population[:,4]) imag(population[:,4]) real(U[1,4]).*cos.(max_eig.*x_range) -real(U[1,4]).*sin.(max_eig.*x_range)], xlabel = "t", 
        labels = ["TDVP: real part" "TDVP: imaginary part" "Exact: real part" "Exact: imaginary part"], legend=:topleft)
        # savefig("Dynamics.png")
        err_plot = plot(x_range, [real(population[:,4]).-real(U[1,4]).*cos.(max_eig.*x_range) imag(population[:,4])+real(U[1,4]).*sin.(max_eig.*x_range)], xlabel = "t",
        labels = ["Error: Real Part" "Error: Imaginary Part"])
        savefig("Error.png")
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

# let 
#     h = LinRange(0.1, 0.5, 50)
#     # h_size = (50:-1:5)
#     err = zeros(length(h))
#     arr_number = 1
#     t0 = 0.0
    
#     init = zeros(ComplexF64,2^N)
#     init[1] = 1.0 + 0.0*im 
#     for i in h
            
#         M_init = MPS(init, sites, maxdim = 1)
#         println("Step size #$arr_number")
#         # h = 1/i
#         T = i
#         # steps = Int64((T - t0)*i)
#         init = zeros(ComplexF64,2^N)
#         init[1] = 1.0 + 0.0*im 
#         M, population = tdvp_constant(H, M_init, t0, T, 1)
#         u = exp(-im.*H_mat*T)*init 
#         err[arr_number] = norm(population[end,:] - u)
#         arr_number += 1
#     end
#     # h = 1 ./h_size
#     h2 = h.^2
#     println(h2)
#     e_p = plot(h, [err h2], xlabel = "H_size", ylabel = "Error", labels = ["Error" L"h^2"], legend=:topleft, title = "Bond Dimension = 1")
#     display(e_p)
#     savefig(e_p, "StepSizeBondDim1.png")
# end

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