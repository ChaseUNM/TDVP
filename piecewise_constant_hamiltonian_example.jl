using ITensors, ITensorMPS
using Plots, Random
using NPZ

include("hamiltonian.jl")
include("tdvp.jl")


N = 2
sites = siteinds("Qubit", N)
freq_01 = [4.80595, 4.8601]
freq_01 .*= 2*pi 

cross_kerr = [0 0 ; 0 0]
Jkl = [0 0.005; 0 0]
Jkl .*= 2*pi 


T = 100.0
t0 = 0.0
steps = 1000
splines = 2
fill_int = Int64(steps/splines)


function repeat_elements(arr, n)
    """
    Repeats each element in the array `arr` exactly `n` times.
    
    Parameters:
        arr (Vector): The input array of numbers.
        n (Int): The number of repetitions for each element.
    
    Returns:
        Vector: The array with repeated elements.
    """
    vcat([fill(x, n) for x in arr]...)
end

Random.seed!(42)

# pt_list = rand(splines)*5
# qt_list = rand(splines)*5

# pt0 = hcat(repeat_elements(pt_list, fill_int), repeat_elements(qt_list, fill_int))'
# qt0 = hcat(repeat_elements(qt_list, fill_int), repeat_elements(qt_list, fill_int))'

pt0 = hcat(vcat(fill(2, fill_int), fill(2.5, fill_int)), vcat(fill(3, fill_int), fill(3.5, fill_int)))'
pt0 = hcat(vcat(fill(-2, fill_int), fill(-2.5, fill_int)), vcat(fill(-3, fill_int), fill(-3.5, fill_int)))'
#Set initial condition to be [0 1 0 0]
init = zeros(ComplexF64,2^N)
init[2] = 1.0 + 0.0*im 
M_init = MPS(init, sites)


function H_t(i)
    H = time_MPO_param(i, pt0, qt0, freq_01, cross_kerr, Jkl, N, sites)
    return H 
end



M_n, population = tdvp_time(H_t, M_init, t0, T, steps)
g1 = plot(range(0, steps).*(T/steps), abs2.(population))
display(g1)

M_init = MPS(init, sites)

M_n_2, population2 = tdvp2_time(H_t, M_init, t0, T, steps)

test_tdvp = false

let
    storage_arr = zeros(ComplexF64, (steps + 1, Int64(2^N)))
    storage_arr[1,:] = init
    step_size = (T - t0)/steps
    
    for i in 1:steps 
        H_mat = matrix_form(H_t(i), sites)
        u = exp(-im.*H_mat.*step_size)*init
        init .= u
        storage_arr[i + 1,:] = init
    end
    
    #Will plot the errors between both the tdvp methods and matrix exponential time stepping
    if test_tdvp == true
        y1 = plot(range(0, steps).*(T/steps), norm.(storage_arr - population), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 1TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"])
        y2 = plot(range(0, steps).*(T/steps), norm.(storage_arr - population2), xlabel = "t", ylabel = "error", 
        plot_title = "Error: 2TDVP and Matrix Exponentiation",labels = ["|00>" "|01>" "|10>" "|11>"], reuse = false)
        plots = plot(y1, y2)
        display(plots)
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


