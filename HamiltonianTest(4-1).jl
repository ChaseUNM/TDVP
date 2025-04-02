using ITensors, ITensorMPS 
using LinearAlgebra, Plots, NPZ, LaTeXStrings
using Plots.PlotMeasures
gr()
include("hamiltonian.jl")
include("tdvp.jl")
N = 3
d = 3
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601, 5.12]*(2*pi)
rot_freq = sum(ground_freq)/N*ones(N)
dipole = [0 0.005 0; 0 0 0.005; 0 0 0]*(2*pi)
self_kerr = [0.2, 0.2, 0.2]*(2*pi)
cross_kerr = zeros(3, 3)

H_s = H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)

function H_t_MPO(t)
    return piecewise_H_MPO_v2(t, pt_correct_unit, qt_correct_unit, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
end

function H_t_MPO_backwards(t)
    return piecewise_H_MPO_v2(t, reverse(pt_correct_unit, dims = 2), reverse(qt_correct_unit, dims = 2), ground_freq, rot_freq, cross_kerr, dipole, N, sites)
end


pt = npzread("3Qubit_spline100_pt.npy")
qt = npzread("3Qubit_spline100_qt.npy")

pcof = npzread("pcof_no_guard.npy")
function construct_pulse(pcof, N) 
    num_par = length(pcof)
    par_per_system = Int64(num_par/(2*N))
    
    pt = zeros(N, par_per_system)
    qt = zeros(N, par_per_system)
    pt[1,:] = pcof[1:par_per_system]
    pt[2,:] = pcof[2*par_per_system + 1:3*par_per_system]
    qt[1,:] = pcof[par_per_system + 1:2*par_per_system]
    qt[2,:] = pcof[3*par_per_system + 1:4*par_per_system]
    pt = pt
    qt = qt
    return pt, qt 
end
# pcof_re, pcof_im = construct_pulse(pcof, N) 
splines = 100

t0 = 0
T = 300.0



pt_correct_unit = pt.*(pi/500)
qt_correct_unit = qt.*(pi/500)
splines = Int64(length(unique(pt_correct_unit))/N)
steps = splines
step_size = (T - t0)/steps
pts = size(pt_correct_unit)[2]
step_size_list = [(count(==(i), qt_correct_unit[1,:])*T/(pts)) for i in unique(qt_correct_unit[1,:])]

times_list = cumsum(vcat(0, step_size_list))

let 
    
    pulse1 = plot(range(0, (pts-1))*T/(pts-1), [pt[1,:] qt[1,:]], ylabel = "MHz", xlabel = "t", labels = ["p(t)" "q(t)"], title = "Qubit 1", dpi = 150)
    pulse2 = plot(range(0, (pts-1))*T/(pts-1), [pt[2,:] qt[2,:]], ylabel = "MHz", xlabel = "t", labels = ["p(t)" "q(t)"], title = "Qubit 2", dpi = 150)
    pulses_plot = plot(pulse1, pulse2, layout = (2, 1))
    # display(pulses_plot)
    # savefig(pulses_plot, "Pulses_plot.png")
end

function smallerize(M)
    row, col = size(M)
    new_col = Int64(length(unique(pt_correct_unit))/N)
    M_n = zeros(row, new_col)
    for i = 1:row
        M_n[i,:] = [(i) for i in unique(M[i,:])]
    end
    return M_n 
end

pt_correct_unit = smallerize(pt_correct_unit)
qt_correct_unit = smallerize(qt_correct_unit)

#Evolve forward and backwards
init = zeros(ComplexF64, d^N)
cutoff = 1E-15
init[6] = 1.0 + 0.0*im
M_init = MPS(init, sites, maxdim = 1)
M_n, _, bd_f = tdvp2_time(H_t_MPO, M_init, t0, T, steps, cutoff, step_size_list)
M0, _, bd_b = tdvp2_time(H_t_MPO_backwards, M_n, t0, T, steps, cutoff, -(step_size_list))

init_abs2 = abs2.(reconstruct_arr_v2(M_init))
M_n_abs2 = abs2.(reconstruct_arr_v2(M_n))
M0_abs2 = abs2.(reconstruct_arr_v2(M0))
display(norm(init_abs2 - M0_abs2))
# display(plot([bd_f, reverse(bd_b), abs.(bd_f - reverse(bd_b))], labels = ["Bond forward" "Bond backward" "Symmetric Test"]))
function bd_plot_forwardbackward(cutoff_list)
    init_list = [1, 2, 5, 6]
    for j in init_list
        bit_string = lpad(string(j - 1, base = 4), 2, '0')
        init = zeros(ComplexF64, d^N)
        init[j] = 1.0 + 0.0*im
        M_init = MPS(init, sites, maxdim = 1)
        bd_plot = plot(layout = (2,2), dpi = 200)    
        for i in 1:length(cutoff_list)
            
            M_n,_,bd_f = tdvp2_time(H_t_MPO, M_init, t0, T, steps, cutoff_list[i], step_size_list)
            M0, _, bd_b = tdvp2_time(H_t_MPO_backwards, M_n, t0, T, steps, cutoff_list[i], -(reverse(step_size_list)))
            err = norm(abs2.(reconstruct_arr_v2(M0)) - abs2.(reconstruct_arr_v2(M_init)))
            err = round(err, digits = 15)
            plot!(bd_plot[i], [bd_f, reverse(bd_b), abs.(bd_f - reverse(bd_b))], labels = ["Bond forward" "Bond backward" "Symmetric Test"], 
            title = "SVD Cutoff: $(cutoff_list[i]), Err: $err", titlefontsize = 6)
        end 
    plot!(bd_plot, plot_title = "Initial Condition: |$bit_string>", titlefontsize = 7)
    plot!(bd_plot, legendfontsize = 4, legend_background_color=RGBA(1, 1, 1, 0.6), legend=:topleft, xlabel = "Step", ylabel = "Bond Dimension") 
    display(bd_plot)
    # savefig(bd_plot, "bd_plot|$bit_string>")
    end
end 
# bd_plot_forwardbackward([1E-15, 1E-10, 1E-5, 1E-3])




function plot_pop(loc, TDVP = 1, cutoff = 0.0, verbose = false)
    init = zeros(ComplexF64, d^N)
    bit_string = lpad(string(loc - 1, base = d), N, '0')
    init[loc] = 1.0 + 0.0*im
    U_g = Matrix(1.0*I, d^N, d^N)
    U_g[5:6, 5:6] = zeros(2, 2)
    U_g[6,5] = 1.0
    U_g[5,6] = 1.0
    expected_out = U_g*init
    M_out = MPS(expected_out, sites)
    if TDVP == 1
        M_init = MPS(init, sites, maxdim = cutoff)
    else
        M_init = MPS(init, sites, maxdim = 1)
    end
    
    if TDVP == 1
        M_N, population = tdvp_time(H_t_MPO, M_init, t0, T, steps, step_size_list, verbose)
        bd = fill(cutoff, length(times_list))
    elseif TDVP == 2
        M_N, population, bd= tdvp2_time(H_t_MPO, M_init, t0, T, steps, cutoff, step_size_list, verbose)
    end

    
    pop_pl = plot(times_list, abs2.(population), legend =:top, legend_column = 16, legendfontsize = 3, dpi = 200)
    display(pop_pl)
    storage_arr = zeros(ComplexF64, (d^N, steps + 1))
    storage_arr[:,1] = init

    error = zeros(d^N, steps + 1)

    error[:,1] = population[1,:] - storage_arr[:,1]
    
    for i = 1:steps
        if verbose == true 
            println("Step $i")
        end
        H_c = H_ctrl(i, pt_correct_unit, qt_correct_unit, N, d)
        H_tot = H_s + H_c
        init = exp(-im*H_tot*step_size_list[i])*init 
        storage_arr[:,i + 1] = init
        error[:,i + 1] = abs2.(population[i + 1,:]) - abs2.(init)
        # display(abs2.(population[i + 1,:]) - abs2.(init))
        # println(population[i + 1,:] - init)
        # println(population[i + 1,:])
        # println(init)
    end

    fidelity = abs.(inner(M_N, M_out))^2
    error_p = plot(range(0, steps)*(T - t0)/steps, [abs.(error[1,:]) abs.(error[2,:]) abs.(error[3,:]) abs.(error[4,:])], legend =:top, legend_column = 16, legendfontsize = 8, 
    dpi = 200, legend_background_color=RGBA(1, 1, 1, 0.8), titlefont=font(10),
    labels = [L"|00\rangle" L"|01\rangle" L"|10\rangle" L"|11\rangle"], 
    ylabel = "Population Error", xlabel = "t", titlepad = -10)
    ftr = text("Fidelity: $fidelity", :black, :right, 8)
    p = plot(times_list, [abs2.(population[:,1]) abs2.(population[:,2]) abs2.(population[:,5]) abs2.(population[:,6])], legend =:top, legend_column = 16, legendfontsize = 8, 
    dpi = 200, legend_background_color=RGBA(1, 1, 1, 0.8), titlefont=font(10),
    labels = [L"|00\rangle" L"|01\rangle" L"|10\rangle" L"|11\rangle"], 
    ylabel = "Population", xlabel = "t")
    # annotate!((0.6, 1.0), ftr)
    if TDVP == 2
        bd_plot = plot(times_list, bd, ylabel = "Bond Dimension", xlabel = "t", yticks = [1, 2, 3, 4], ylimits = (1,4))
    end
    # display(error_p)
    println("Fidelity starting from |$bit_string>: ", abs.(inner(M_N, M_out))^2)
    pop_end = population[end,:]
    if TDVP == 1
        return p, pop_end, bd
    elseif TDVP == 2
        return p, pop_end, bd
    end
end

# p = plot_pop(1)
function all_plots(TDVP = 1, cutoff = 0.0)
    # loc_arr = [1, 2, 3, 4]
    # rows, cols = 2, 2
    # plt = plot(layout = (rows, cols))
    # for i in 1:(rows*cols)
    #     plot!(plt, plot_pop(i), subplot = i)
    # end
    # display(plt)\
    UT = zeros(ComplexF64, d^N, 2^N)
    i_l = [1, 2, 5, 6]
    println(size(UT))
    for i = 1:length(i_l)
        _, e, _ = plot_pop(i_l[i], TDVP, cutoff)
        UT[:,i] = e 
        println("Norm e: ", norm(e))
    end
    i1 = 1
    i2 = 2
    i3 = 5
    i4 = 6
    #Reduced UT
    p1, e1, bd1 = plot_pop(i1, TDVP, cutoff)
    p2, e2, bd2 = plot_pop(i2, TDVP, cutoff)
    p3, e3, bd3 = plot_pop(i3, TDVP, cutoff)
    p4, e4, bd4 = plot_pop(i4, TDVP, cutoff)
    V = Matrix(1.0*I, d^N, d^N)
    V[i3:i4, i3:i4] = zeros(2, 2)
    V[i4,i3] = 1.0
    V[i3,i4] = 1.0
    e_l = hcat(e1, e2, e3, e4)
    # UT = abs2.(UT)
    Vtg = Matrix(1.0*I, d^N, 2^N)
    Vtg[3:4, 3:4] .= 0
    Vtg[6,3] = 1.0
    Vtg[5,4] = 1.0
    display(abs2.(UT))
    display(Vtg)
    gate_fidelity = abs.(tr(UT'*Vtg))^N/d^N
    println("Gate Fidelity: ", gate_fidelity)
    str = text("Gate Fidelity: $gate_fidelity", 10)
    plt = plot(p1, p2, p3, p4, layout = (2,2), dpi = 250, size = (800, 600), plot_title = "SVD Cutoff: $cutoff")
    for i in i_l
        for j in 1:4
            if abs2.(e_l[i,j]) > 0.0001
                annotate!(plt[j], times_list[end] - 10, abs2.(e_l[i,j]), text("$(round.(abs2.(e_l[i, j]), digits = 5))", 8, :black))
            end
        end 
    end
    annotate!(plt[4], -1, -0.2, str)
    
    # bd_plot = plot(bd1, bd2, bd3, bd4, layout = (2, 2), dpi = 250, size = (800,600))
    
    display(plt)
    println("Press 'Enter' to continue")
    readline()
    # savefig(plt, "TDVP2_Evolution.png")
    # savefig(bd_plot, "BD_TDVP2_2Guard5E-3.png")
end

# all_plots(2, 0.0)

function bond_plots(cutoff_list, TDVP = 2)
    
    V = Matrix(1.0*I, d^N, 2^N)
    V[3:8, 3:8] .= 0
    V[14,7] = 1.0
    V[13,8] = 1.0
    V[11, 6] = 1.0
    V[10, 5] = 1.0
    V[5, 4] = 1.0
    V[4, 3] =1.0
    i_l = [1, 2, 4, 5, 10, 11, 13, 14]
    display(V'*V)
    bd_plot = plot(layout = (4,2), dpi = 400)
    linestyles = [:solid, :dash, :dot, :dashdot, :solid]
    for i in 1:length(cutoff_list)
        fidelity = 0.0
        UT = zeros(ComplexF64, d^N, 2^N)
        for j = 1:length(i_l) 
            _, e, _ = plot_pop(i_l[j], TDVP, cutoff_list[i])
            UT[:,j] = (e)
            display(abs2.(e))
            display(abs2(e[i_l[j]]))
            println(V[j,j])
            fidelity += V[:,j]'*e    
        end
        fidelity = abs(1/length(i_l)*fidelity)^2
        println("Fidelity: ", fidelity)
        println(size(abs.(UT'*V)))
        display(abs.(UT'*V))
        println(abs.(tr(UT'*V)))
        gate_fidelity = (1/2^N)*(abs.(tr(UT'*V)))
        println("Gate Fidelity: ", fidelity)
        display(UT)
        _,_,b1 = plot_pop(1, TDVP, cutoff_list[i])
        _,_,b2 = plot_pop(2, TDVP, cutoff_list[i])
        _,_,b3 = plot_pop(4, TDVP, cutoff_list[i])
        _,_,b4 = plot_pop(5, TDVP, cutoff_list[i])
        _,_,b5 = plot_pop(10, TDVP, cutoff_list[i])
        _,_,b6 = plot_pop(11, TDVP, cutoff_list[i])
        _,_,b7 = plot_pop(13, TDVP, cutoff_list[i])
        _,_,b8 = plot_pop(14, TDVP, cutoff_list[i])
        plot!(bd_plot[1], times_list, b1, xlabel = "t", ylabel = "Bond Dimension", label = "SVD Cutoff: $(cutoff_list[i]) | Gate Fidelity: $fidelity", linestyle=linestyles[i])
        plot!(bd_plot[2], times_list, b2, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[3], times_list, b3, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[4], times_list, b4, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[5], times_list, b5, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[6], times_list, b6, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[7], times_list, b7, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[8], times_list, b8, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
    end
    plot!(bd_plot, legendfontsize = 3, legend_background_color=RGBA(1, 1, 1, 0.6), legend=:topleft) 
    display(bd_plot)
    # savefig(bd_plot, "3Q_bd_plot_TDVP2.png")
end

bond_plots([1E-4], 2)

#OpSum Testing
