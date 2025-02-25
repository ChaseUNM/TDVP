using ITensors, ITensorMPS
using Plots, Random
using NPZ

include("hamiltonian.jl")
include("tdvp.jl")

Random.seed!(42)

N = 2
sites = siteinds("Qubit", N)
freq_01 = [1, 2]
freq_01 .*= 1 

cross_kerr = [0 0 ; 0 0]
Jkl = [0 0.5; 0 0]
Jkl .*= 1 


T = 2.0
t0 = 0.0
steps = 100
splines = 2





# pt0 = hcat(vcat(fill(2, fill_int), fill(2.5, fill_int)), vcat(fill(3, fill_int), fill(3.5, fill_int)))'
# pt0 = hcat(vcat(fill(-2, fill_int), fill(-2.5, fill_int)), vcat(fill(-3, fill_int), fill(-3.5, fill_int)))'
#Set initial condition to be [0 1 0 0]
init = zeros(ComplexF64,2^N)
init[1] = 1.0 + 0.0*im 
M_init = MPS(init, sites, maxdim = 2)

pt0 = [1 2; 1.5 2.5]
qt0 = [-1 -2; -1.5 -2.5]

pt_pulse, qt_pulse = downsample_pulse(pt0, qt0, splines, steps)

function H_t_MPO(i)
    H = piecewise_H_MPO_no_rot(i, pt_pulse, freq_01, cross_kerr, Jkl, N, sites)
    return H 
end

function H_t(i)
    H = piecewise_H_no_rot(i, pt_pulse, freq_01, cross_kerr, Jkl, N)
    return H 
end


let
    false_count = 0
    for i in 1:steps 
        H_mat_test = H_t(i)
        if ishermitian(H_mat_test) == false
            false_count += 1
        end
    end
    println("We have $false_count times where the Hamiltonian is not hermitian")
end


M_n, population = tdvp_time(H_t_MPO, M_init, t0, T, steps)
# g1 = plot(range(0, steps).*(T/steps), abs2.(population))
# display(g1)

M_init = MPS(init, sites)

M_n_2, population2 = tdvp2_time(H_t_MPO, M_init, t0, T, steps, 0)

test_tdvp = true

let
    if test_tdvp == true
        storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
        storage_arr[1,:] = init
        step_size = (T - t0)/steps
        
        for i in 1:steps 
            H_mat = H_t(i)
            u = exp(-im.*H_mat.*step_size)*init
            init .= u
            storage_arr[i + 1,:] = init
        end
        
        #Will plot the errors between both the tdvp methods and matrix exponential time stepping
        
        y1 = plot(range(0, steps).*(T/steps), norm.(storage_arr - population), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 1TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"], dpi = 150)
        y2 = plot(range(0, steps).*(T/steps), norm.(storage_arr - population2), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 2TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"], reuse = false, dpi = 150)
        plots = plot(y1, y2)
        sol = plot(range(0, steps).*(T/steps), abs2.(storage_arr), xlabel = "t", ylabel = "Population", plot_title = "Matrix Exponentation Evolution",
        labels = ["|00>" "|01>" "|10>" "|11>"], reuse = false, dpi = 150)
        savefig(y1, "PiecewiseConstantTDVP.png")
        savefig(y2, "PiecewiseConstantTDVP2.png")
        savefig(sol, "PiecewiseConstantEvolution.png")
    end
end




# let
#     for N in 2:8 
#         sites = siteinds("Qubit", N)
        
#         function H_t(i)
#             H = time_MPO_param(i, pt0, qt0, freq_01, cross_kerr, Jkl, N, sites)
#             return H 
#         end
#         T = 1.0
#         t0 = 0.0
#         steps = 100
        


#         init = zeros(ComplexF64,2^N)
#         init[1] = 1.0 + 0.0*im
#         M_init = MPS(init, sites)
#         storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
#         storage_arr[1,:] = init
#         step_size = (T - t0)/steps

#         for i in 1:steps 
#             H_mat = matrix_form(H_t(i), sites)
#             u = exp(-im.*H_mat.*step_size)*init
#             init .= u
#             storage_arr[i + 1,:] = init
#         end 
        
#         println("Linkdims before: ", linkdims(M_init))
#         t0 = 0.0
#         T = 1.0
#         steps = 1
        
#         M_N, population = tdvp_time(H_t, M_init, t0, T, steps)
        
#         println(norm(population[end,:] - storage_arr[end,:]))
#         println("Linkdims after: ", linkdims(M_N))
#     end
# end


