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

pt = npzread("2Qubit_bspline_pt.npy")
qt = npzread("2Qubit_bspline_qt.npy")

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
pcof_re, pcof_im = construct_pulse(pcof, N) 
splines = size(pcof_re)[2]

t0 = 0
T = 300.0



pt_correct_unit = pt.*(pi/500)
qt_correct_unit = qt.*(pi/500)
# splines = Int64(length(unique(pt_correct_unit))/2)

steps = length(pt_correct_unit[1,:])
# steps = 20
println(steps)
step_size = (T - t0)/steps
pts = size(pt_correct_unit)[2]
step_size_list = [(count(==(i), qt_correct_unit[1,:])*T/(pts)) for i in unique(qt_correct_unit[1,:])]

times_list = cumsum(vcat(0, step_size_list))

let 
    
    pulse1 = plot(range(0, (pts-1))*T/(pts-1), [pt[1,:] qt[1,:]], ylabel = "MHz", xlabel = "t", labels = ["p(t)" "q(t)"], title = "Qubit 1", dpi = 150)
    pulse2 = plot(range(0, (pts-1))*T/(pts-1), [pt[2,:] qt[2,:]], ylabel = "MHz", xlabel = "t", labels = ["p(t)" "q(t)"], title = "Qubit 2", dpi = 150)
    pulses_plot = plot(pulse1, pulse2, layout = (2, 1))
    display(pulses_plot)
    # savefig(pulses_plot, "B_Spline_pulses_plot.png")
end

function smallerize(M)
    row, col = size(M)
    new_col = Int64(length(unique(pt_correct_unit))/2)
    M_n = zeros(row, new_col)
    for i = 1:row
        M_n[i,:] = [(i) for i in unique(M[i,:])]
    end
    return M_n 
end

# pt_correct_unit = smallerize(pt_correct_unit)
# qt_correct_unit = smallerize(qt_correct_unit)


function plot_pop(loc, TDVP = 1, cutoff = 0.0, verbose = false)
    init = zeros(ComplexF64, d^N)
    bit_string = lpad(string(loc - 1, base = 4), 2, '0')
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
        M_N, population = tdvp_time(H_t_MPO, M_init, t0, T, steps, [], true)
        bd = fill(cutoff, length(times_list))
    elseif TDVP == 2
        M_N, population, bd = tdvp2_time(H_t_MPO, M_init, t0, T, steps, cutoff)
    end

    # Bond Dimension
    # pop_pl = plot(LinRange(0, T, steps), abs2.(population), legend =:top, legend_column = 16, legendfontsize = 3, dpi = 200)
    # display(pop_pl)
    storage_arr = zeros(ComplexF64, (d^N, steps + 1))
    storage_arr[:,1] = init

    error = zeros(d^N, steps + 1)

    error[:,1] = population[1,:] - storage_arr[:,1]
    
    for i = 1:steps 
        println("Step $i")
        H_c = H_ctrl(i, pt_correct_unit, qt_correct_unit, N, d)
        H_tot = H_s + H_c
        init = exp(-im*H_tot*step_size)*init 
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
    p = plot(LinRange(0, T, steps + 1), [abs2.(population[:,1]) abs2.(population[:,2]) abs2.(population[:,5]) abs2.(population[:,6])], legend =:top, legend_column = 16, legendfontsize = 8, 
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
    if TDVP == 1
        str_def = string("Bond Dimension: ")
    else
        str_def = string("SVD Cutoff: ")
    end
    plt = plot(p1, p2, p3, p4, layout = (2,2), dpi = 250, size = (800, 600), plot_title = string(str_def, cutoff))
    for i in i_l
        for j in 1:4
            if abs2.(e_l[i,j]) > 0.0001
                annotate!(plt[j], times_list[end] - 10, abs2.(e_l[i,j]), text("$(round.(abs2.(e_l[i, j]), digits = 5))", 8, :black))
            end
        end 
    end
    annotate!(plt[4], -1, -0.2, str)
    
    # bd_plot = plot(bd1, bd2, bd3, bd4, layout = (2, 2), dpi = 250, size = (800,600))
    
    # display(plt)
    # println("Press 'Enter' to continue")
    # readline()
    title = string("TDVP$(TDVP)_Evolution$str_def$cutoff.png")
    savefig(plt, title)
    # savefig(bd_plot, "BD_TDVP2_2Guard5E-3.png")
end

all_plots(1, 4)
all_plots(1, 3)
all_plots(1, 2)
all_plots(1, 1)
all_plots(2, 1E-10)
all_plots(2, 1E-5)
all_plots(2, 1E-3)
all_plots(2, 1E-2)

function bond_plots(cutoff_list, TDVP = 2)
    
    V = Matrix(1.0*I, d^N, 2^N)
    V[3:4, 3:4] .= 0
    V[6,3] = 1.0
    V[5,4] = 1.0
    i_l = [1, 2, 5, 6]
    bd_plot = plot(layout = (2,2), dpi = 200)
    linestyles = [:solid, :dash, :dot, :dashdot, :solid]
    for i in 1:length(cutoff_list)
        UT = zeros(ComplexF64, d^N, 2^N)
        for j = 1:length(i_l) 
            _, e, _ = plot_pop(i_l[j], TDVP, cutoff_list[i])
            UT[:,j] = e
        end
        
        gate_fidelity = round(1/d^N*(abs.(tr(UT'*V))^N), digits = 5, RoundDown)
        println("Gate Fidelity: ", gate_fidelity)
        display(UT)
        _,_,b1 = plot_pop(1, TDVP, cutoff_list[i], true)
        _,_,b2 = plot_pop(2, TDVP, cutoff_list[i], true)
        _,_,b3 = plot_pop(5, TDVP, cutoff_list[i], true)
        _,_,b4 = plot_pop(6, TDVP, cutoff_list[i], true)
        plot!(bd_plot[1], LinRange(0,T,steps + 1), b1, xlabel = "t", ylabel = "Bond Dimension", label = "SVD Cutoff: $(cutoff_list[i]) | Gate Fidelity: $gate_fidelity", linestyle=linestyles[i])
        plot!(bd_plot[2], LinRange(0,T,steps + 1), b2, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[3], LinRange(0,T,steps + 1), b3, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
        plot!(bd_plot[4], LinRange(0,T,steps + 1),  b4, label = "SVD Cutoff: $(cutoff_list[i])", linestyle=linestyles[i])
    end
    plot!(bd_plot, legendfontsize = 4, legend_background_color=RGBA(1, 1, 1, 0.6), legend=:topleft) 
    display(bd_plot)
    savefig(bd_plot, "bd_plot_TDVP2.png")
end

bond_plots([0.0, 1E-10, 1E-7, 1E-5, 1E-3], 2)

#OpSum Testing
