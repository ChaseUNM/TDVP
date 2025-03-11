using ITensors, ITensorMPS 
using LinearAlgebra, Plots, NPZ, LaTeXStrings
using Plots.PlotMeasures


gr()
include("hamiltonian.jl")
include("tdvp.jl")
N = 2
d = 4
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601]*(2*pi)
rot_freq = sum(ground_freq)/N*ones(N)
dipole = [0 0.005; 0 0]*(2*pi)
self_kerr = [0.2, 0.2]*(2*pi)
cross_kerr = zeros(2, 2)

H_s = H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)

function H_t_MPO(t)
    return piecewise_H_MPO_v2(t, pt_correct_unit, qt_correct_unit, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
end

pt = npzread("pt_guard_2.npy")
qt = npzread("qt_guard_2.npy")

pt_q1 = pt[1,:]
pt_q2 = pt[2,:]
qt_q1 = qt[1,:]
qt_q2 = qt[2,:]

pt_correct_unit = pt.*(pi/500)
qt_correct_unit = qt.*(pi/500)

println(size(pt))

t0 = 0
T = 300.0
steps = 54132
step_size = (T - t0)/steps


H_mat = matrix_form(H_t_MPO(1), sites)
H = H_s + H_ctrl(1, pt_correct_unit, qt_correct_unit, N, d)


# let 
#     init = zeros(ComplexF64, d^N)
#     init[2] = 1.0 + 0.0*im
#     M_init = MPS(init, sites, maxdim = 1)
#     M_N, population, bd = tdvp2_time(H_t_MPO, M_init, t0, T, steps, 1E-12)
    
#     pop_pl = plot((range(0, steps)*(T - t0)/steps, abs2.(population)), legend =:top, legend_column = 16, legendfontsize = 3, dpi = 200)
#     display(pop_pl)
#     storage_arr = zeros(ComplexF64, (d^N, steps + 1))
#     storage_arr[:,1] = init

#     error = zeros(d^N, steps + 1)
#     println(size(population[:,1]))
#     println(size(storage_arr[:,1]))
#     error[:,1] = population[1,:] - storage_arr[:,1]
    
#     for i = 1:steps 
#         println("Step $i")
#         H_c = H_ctrl(i, pt_correct_unit, qt_correct_unit, N, d)
#         H_tot = H_s + H_c
#         init = exp(-im*H_tot*step_size)*init 
#         storage_arr[:,i + 1] = init
#         error[:,i + 1] = abs2.(population[i + 1,:]) - abs2.(init)
#         # println(population[i + 1,:] - init)
#         # println(population[i + 1,:])
#         # println(init)
#     end


#     println(error[:,end])
#     error_p = plot(range(0, steps)*(T - t0)/steps, bd)
#     println(abs2.(population[end,:]))
#     println(abs2.(storage_arr[:,end]))
#     p = plot(range(0, steps)*(T - t0)/steps, abs2.(storage_arr'), legend =:top, legend_column = 16, legendfontsize = 6, 
#     dpi = 200, legend_background_alpha = 0.5)
#     # labels = [L"|00\rangle" L"|01\rangle" L"|02\rangle" L"|10\rangle" L"|11\rangle" L"|12\rangle" L"|20\rangle" L"|21\rangle" L"|22\rangle"])
#     # display(error_p)
#     display(p)
# end

function plot_pop(loc)
    init = zeros(ComplexF64, d^N)
    bit_string = lpad(string(loc - 1, base = 4), 2, '0')
    init[loc] = 1.0 + 0.0*im
    M_init = MPS(init, sites)
    M_N, population= tdvp_time(H_t_MPO, M_init, t0, T, steps)
    
    pop_pl = plot((range(0, steps)*(T - t0)/steps, abs2.(population)), legend =:top, legend_column = 16, legendfontsize = 3, dpi = 200)
    display(pop_pl)
    storage_arr = zeros(ComplexF64, (d^N, steps + 1))
    storage_arr[:,1] = init

    error = zeros(d^N, steps + 1)
    println(size(population[:,1]))
    println(size(storage_arr[:,1]))
    error[:,1] = population[1,:] - storage_arr[:,1]
    
    for i = 1:steps 
        println("Step $i")
        H_c = H_ctrl(i, pt_correct_unit, qt_correct_unit, N, d)
        H_tot = H_s + H_c
        init = exp(-im*H_tot*step_size)*init 
        storage_arr[:,i + 1] = init
        error[:,i + 1] = abs2.(population[i + 1,:]) - abs2.(init)
        # println(population[i + 1,:] - init)
        # println(population[i + 1,:])
        # println(init)
    end


    println(error[:,end])
    error_p = plot(range(0, steps)*(T - t0)/steps, [abs.(error[1,:]) abs.(error[2,:]) abs.(error[5,:]) abs.(error[6,:])], legend =:top, legend_column = 16, legendfontsize = 8, 
    dpi = 200, legend_background_color=RGBA(1, 1, 1, 0.8), titlefont=font(10),
    labels = [L"|00\rangle" L"|01\rangle" L"|10\rangle" L"|11\rangle"], 
    ylabel = "Population Error", xlabel = "t", titlepad = -10)
    println(abs2.(population[end,:]))
    println(abs2.(storage_arr[:,end]))
    p = plot(range(0, steps)*(T - t0)/steps, [abs2.(population[:,1]) abs2.(population[:,2]) abs2.(population[:,5]) abs2.(population[:,6])], legend =:top, legend_column = 16, legendfontsize = 8, 
    dpi = 200, legend_background_color=RGBA(1, 1, 1, 0.8), titlefont=font(10),
    labels = [L"|00\rangle" L"|01\rangle" L"|10\rangle" L"|11\rangle"], 
    ylabel = "Population", xlabel = "t", titlepad = -10)
    # bd_plot = plot(range(0, steps)*(T - t0)/steps, bd, ylabel = "Bond Dimension", xlabel = "t")
    # display(error_p)
    return error_p
end


# p = plot_pop(1)
let 
    # loc_arr = [1, 2, 3, 4]
    # rows, cols = 2, 2
    # plt = plot(layout = (rows, cols))
    # for i in 1:(rows*cols)
    #     plot!(plt, plot_pop(i), subplot = i)
    # end
    # display(plt)
    p1 = plot_pop(1)
    p2 = plot_pop(2)
    p3 = plot_pop(5)
    p4 = plot_pop(6)
    plt = plot(p1, p2, p3, p4, layout = (2,2), dpi = 250, size = (800, 600))
    # bd_plot = plot(bd1, bd2, bd3, bd4, layout = (2, 2), dpi = 250, size = (800,600))
    display(plt)
    savefig(plt, "2GuardTDVPError.png")
    # savefig(bd_plot, "BD_TDVP2_2Guard1E-15.png")
end
#OpSum Testing



H_MPO = piecewise_H_MPO_v2(2, pt_correct_unit, qt_correct_unit, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
H_MPO_mat = matrix_form(H_MPO, sites)
display(H_MPO_mat)
H_c = H_ctrl(2, pt_correct_unit, qt_correct_unit, N, d)
H_mat = H_s + H_ctrl(2, pt_correct_unit, qt_correct_unit, N, d)

display(H_mat)
init = zeros(ComplexF64, d^N)

init[1] = 1.0 + 0.0*im




tdvp_p = plot(range(0, steps)*(T - t0)/steps, abs2.(population), legend =:top, legend_column = 9, legendfontsize = 5, dpi = 200,
labels = [L"|00\rangle" L"|01\rangle" L"|02\rangle" L"|10\rangle" L"|11\rangle" L"|12\rangle" L"|20\rangle" L"|21\rangle" L"|22\rangle"])

bd = plot(range(0, steps)*(T - t0)/steps, bd_list)

# display(bd)
# println(size(population))

println(abs2.(population[end,:]))