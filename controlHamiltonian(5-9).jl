using ITensorMPS, ITensors 
using LinearAlgebra, Plots, NPZ, DelimitedFiles

gr()
include("hamiltonian(5-5).jl")

N = 2
d = 4
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601]*(2*pi)
# ground_freq = [1,2]
rot_freq = sum(ground_freq)/N*ones(N)  
# rot_freq = [0,0] 
dipole = [0 0.005; 0 0]*(2*pi)
# dipole = zeros(2,2)
# self_kerr = [0.2, 0.2]*(2*pi)
self_kerr = [0,0]
# self_kerr = [2,2]
cross_kerr = zeros(2, 2)

om = zeros(2,2)
om[1,1] = (ground_freq[1] - rot_freq[1])
om[1,2] = (ground_freq[2] - rot_freq[2])
om[2,1] = (ground_freq[1] - rot_freq[1] - self_kerr[1])
om[2,2] = (ground_freq[2] - rot_freq[2] - self_kerr[2])

om[1,1] = 0.027532809972830558*2*pi
om[1,2] = -0.027532809972830558*2*pi 
om[2,1] = 0.027532809972830558*2*pi 
om[2,2] = -0.027532809972830558*2*pi


# pt = npzread("2Qubit_bspline_pt.npy")
# qt = npzread("2Qubit_bspline_qt.npy")

pt = npzread("2Qubit_bspline_pt2.npy").*(pi/500)
qt = npzread("2Qubit_bspline_qt2.npy").*(pi/500)

H = H_MPO_v2(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)

H_s = H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)

H_c = piecewise_H_MPO_v2(2, pt, qt, ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)


H1_arr = Array(H[1], inds(H[1]))
H2_arr = Array(H[2], inds(H[2]))

H_c1_arr = Array(H_c[1], inds(H_c[1]))
H_c2_arr = Array(H_c[2], inds(H_c[2]))

println(pt[:,2])
println(qt[:,2])

display(H1_arr[1,:,:])
display(H_c1_arr[1,:,:])


function annihilation_operator(N::Int)
    a = zeros(ComplexF64, N, N)
    for n in 2:N
        a[n-1, n] = sqrt(n - 1)
    end
    return a
end

struct bcparams
    T ::Float64
    D1::Int64 # number of B-spline coefficients per control function
    om::Array{Float64,2} #Carrier wave frequencies [rad/s], size Nfreq
    tcenter::Array{Float64,1}
    dtknot::Float64
    pcof::Array{Float64,1} # coefficients for all 2*Ncoupled splines, size Ncoupled*D1*Nfreq*2 (*2 because of sin/cos)
    Nfreq::Int64 # Number of frequencies
    Ncoeff:: Int64 # Total number of coefficients
    Ncoupled::Int64 # Number of B-splines functions for the coupled ctrl Hamiltonians
    Nunc::Int64 # Number of B-spline functions  for the UNcoupled ctrl Hamiltonians

    # New constructor to allow defining number of symmetric Hamiltonian terms
    function bcparams(T::Float64, D1::Int64, Ncoupled::Int64, Nunc::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
        dtknot = T/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)
        Nfreq = size(omega,2)
        nCoeff = Nfreq*D1*2*(Ncoupled + Nunc)
        if nCoeff != length(pcof)
            println("nCoeff = ", nCoeff, " Nfreq = ", Nfreq, " D1 = ", D1, " Ncoupled = ", Ncoupled, " Nunc = ", Nunc, " len(pcof) = ", length(pcof))
            throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
        end
        new(T, D1, omega, tcenter, dtknot, pcof, Nfreq, nCoeff, Ncoupled, Nunc)
    end

end

# simplified constructor (assumes no uncoupled terms)
function bcparams(T::Float64, D1::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
  dtknot = T/(D1 -2)
  tcenter = dtknot.*(collect(1:D1) .- 1.5)
  Ncoupled = size(omega,1) # should check that Ncoupled >=1
  Nfreq = size(omega,2)
  Nunc = 0
  nCoeff = Nfreq*D1*2*Ncoupled
  if nCoeff != length(pcof)
    throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
  end
  bcparams(T, D1, Ncoupled, Nunc, omega, pcof)
end

"""
    f = bcarrier2(t, params, func)

Evaluate a B-spline function with carrier waves. See also the `bcparams` constructor.

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::params`: Parameters for the spline
- `func::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function bcarrier2(t::Float64, params::bcparams, func::Int64)
    # for a single oscillator, func=0 corresponds to p(t) and func=1 to q(t)
    # in general, 0 <= func < 2*Ncoupled + Nunc

    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = div(func, 2) # osc is base 0; 0<= osc < Ncoupled
    q_func = func % 2 # q_func = 0 for p and q_func=1 for q
    
    f = 0.0 # initialize
    
    dtknot = params.dtknot
    width = 3*dtknot
    
    k = max.(3, ceil.(Int64,t./dtknot + 2)) # pick out the index of the last basis function corresponding to t
    k = min.(k, params.D1) #  Make sure we don't access outside the array
    
    if func < 2*(params.Ncoupled + params.Nunc)
        # Coupled and uncoupled controls
        @fastmath @inbounds @simd for freq in 1:params.Nfreq
            fbs1 = 0.0 # initialize
            fbs2 = 0.0 # initialize
            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc
            offset1 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1
            offset2 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1 + params.D1

            # 1st segment of nurb k
            tc = params.tcenter[k]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            fbs2 += params.pcof[offset2+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            
            # 2nd segment of nurb k-1
            tc = params.tcenter[k-1]
            tau = (t - tc)./width
            fbs1 += params.pcof[offset1+k.-1] .* (0.75 - 9 *tau^2)
            fbs2 += params.pcof[offset2+k.-1] .* (0.75 - 9 *tau^2)
            
            # 3rd segment of nurb k-2
            tc = params.tcenter[k.-2]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
            fbs2 += params.pcof[offset2+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)

            #    end # for carrier phase
            # p(t)
            if q_func==1
                f += fbs1 * sin(params.om[osc+1,freq]*t) + fbs2 * cos(params.om[osc+1,freq]*t) # q-func
            else
                f += fbs1 * cos(params.om[osc+1,freq]*t) - fbs2 * sin(params.om[osc+1,freq]*t) # p-func
            end
        end # for freq
    end # if
    return f
end

params = reshape(readdlm("params2.dat"), 160)
bc_params = bcparams(300.0,20, om, params)


pt = npzread("2Qubit_bspline_pt2.npy").*(pi/500)
qt = npzread("2Qubit_bspline_qt2.npy").*(pi/500)




function H_MPO_control(ground_freq, rot_freq, self_kerr, cross_kerr, bcparams, t, dipole, N, sites)
    #Construct Hamiltonian Manually with no control 

    pt_1 = bcarrier2(t, bcparams, 0)
    qt_1 = bcarrier2(t, bcparams, 1)
    pt_2 = bcarrier2(t, bcparams, 2)
    qt_2 = bcarrier2(t, bcparams, 3)

    H = MPO(N)
    
    s1 = dim(sites[1])
    s2 = dim(sites[2])
    a1 = annihilation_operator(s1)
    a2 = annihilation_operator(s2)
    l1 = 4 
    H1 = zeros(ComplexF64, s1, s1, l1)
    H2 = zeros(ComplexF64, s2, s2, l1)
    H1[1,:,:] = (ground_freq[2] - rot_freq[2])*(a1'*a1) - 0.5*self_kerr[2]*(a1'*a1'*a1*a1) + pt_2*(a1 + a1') + im*qt_2*(a1 - a1')
    H1[2,:,:] = dipole[1,2]*a1'
    H1[3,:,:] = dipole[1,2]*a1
    H1[4,:,:] = Matrix(1.0*I, s1, s1)
    
    H2[1,:,:] = Matrix(1.0*I,  s2, s2)
    H2[2,:,:] = a2
    H2[3,:,:] = a2'
    H2[4,:,:] = (ground_freq[1] - rot_freq[1])*(a2'*a2) - 0.5*self_kerr[1]*(a2'*a2'*a2*a2) + pt_1*(a2 + a2') + im*qt_1*(a1 - a1')
    s1 = sites[1]
    s2 = sites[2]
    l1 = Index(l1, "Link, l = 1")
    
    H[1] = ITensor(H1, l1, s1, s1')
    H[2] = ITensor(H2, l1, s2, s2')

    return H 
end


function H_MPO_manual(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
    #Construct Hamiltonian Manually with no control 

    H = MPO(ComplexF64, sites)
    
    s1 = dim(sites[1])
    s2 = dim(sites[2])
    a1 = annihilation_operator(s1)
    a2 = annihilation_operator(s2)
    l1 = 4 
    H1 = zeros(ComplexF64, s1, s1, l1)
    H2 = zeros(ComplexF64, s2, s2, l1)
    H1[1,:,:] = (ground_freq[2] - rot_freq[2])*(a1'*a1) - 0.5*self_kerr[2]*(a1'*a1'*a1*a1)
    H1[2,:,:] = dipole[1,2]*a1'
    H1[3,:,:] = dipole[1,2]*a1
    H1[4,:,:] = Matrix(1.0*I, s1, s1)
    
    H2[1,:,:] = Matrix(1.0*I,  s2, s2)
    H2[2,:,:] = a2
    H2[3,:,:] = a2'
    H2[4,:,:] = (ground_freq[1] - rot_freq[1])*(a2'*a2) - 0.5*self_kerr[1]*(a2'*a2'*a2*a2)
    s1 = sites[1]
    s2 = sites[2]
    l1 = Index(l1, "Link, l = 1")
    
    H[1] = ITensor(H1, l1, s1', s1)
    H[2] = ITensor(H2, l1, s2', s2)

    return H 
end

function update_H(H_MPO::MPO, bcparams, t)
    pt_1 = bcarrier2(t, bcparams, 0)
    qt_1 = bcarrier2(t, bcparams, 1)
    pt_2 = bcarrier2(t, bcparams, 2)
    qt_2 = bcarrier2(t, bcparams, 3)

    # println(H_MPO)
    
    links = linkinds(H_MPO)
    site1_inds = siteinds(H_MPO)[1]
    site2_inds = siteinds(H_MPO)[2]

    site1 = H_MPO[1]
    site2 = H_MPO[2]
    # println(site2)
    # display(Array(H_MPO[1], inds(H_MPO[1]))[1,:,:])
    # display(Array(site2, inds(site2))[4,:,:])
    for i = 1:4
        if i < 4
            H_MPO[1][links[1] => 1, site1_inds[1] => i, site1_inds[2] => i + 1] = sqrt(i)*(pt_2 + im*qt_2)
            H_MPO[2][links[1] => 4, site2_inds[1] => i, site2_inds[2] => i + 1] = sqrt(i)*(pt_1 + im*qt_1)
        end
        if i > 1
            H_MPO[1][links[1] => 1, site1_inds[1] => i, site1_inds[2] => i - 1] = sqrt(i - 1)*(pt_2 - im*qt_2)
            H_MPO[2][links[1] => 4, site2_inds[1] => i, site2_inds[2] => i - 1] = sqrt(i - 1)*(pt_1 - im*qt_1)
        end
    end
    println("-----------------------------------")
    # site1[links[1] => 1, site1_inds[1] => 1, site1_inds[2] => 2] = 1
    # site1[links[1] => 1, site1_inds[1] => 2, site1_inds[2] => 1] = 1
    # site1[links[1] => 1, site1_inds[1] => 2, site1_inds[2] => 3] = sqrt(2)
    # site1[links[1] => 1, site1_inds[1] => 3, site1_inds[2] => 2] = sqrt(2)
    # site1[links[1] => 1, site1_inds[1] => 3, site1_inds[2] => 4] = sqrt(3)
    # site1[links[1] => 1, site1_inds[1] => 4, site1_inds[2] => 3] = sqrt(3)
    # display(Array(H_MPO[1], inds(H_MPO[1]))[1,:,:])

    # display(Array(site2, inds(site2))[4,:,:])
    # site2[links[1] => 4, site2_inds[1] => 1, site2_inds[2] => 2] = 1
    # site2[links[1] => 4, site2_inds[1] => 2, site2_inds[2] => 1] = 1
    # site2[links[1] => 4, site2_inds[1] => 2, site2_inds[2] => 3] = sqrt(2)
    # site2[links[1] => 4, site2_inds[1] => 3, site2_inds[2] => 2] = sqrt(2)
    # site2[links[1] => 4, site2_inds[1] => 3, site2_inds[2] => 4] = sqrt(3)
    # site2[links[1] => 4, site2_inds[1] => 4, site2_inds[2] => 3] = sqrt(3)
    return H_MPO
end

H = H_MPO_manual(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
pts = 54132
t_list = LinRange(0,300,pts)
H_c2 = H_MPO_control(ground_freq, rot_freq, self_kerr, cross_kerr, bc_params, t_list[3], dipole, N, sites)
println("----------------------------")
display(Array(H_c2[1], inds(H_c2[1]))[1,:,:])
println("----------------------------")
display(Array(H_c[1], inds(H_c[1]))[1,:,:])

f_eval = zeros(pts)

for i = 1:pts 
    f_eval[i] = bcarrier2(t_list[i], bc_params, 0)*(500/pi)
end


for j = 1:4 
    display(norm(Array(H_c2[1], inds(H_c2[1]))[j,:,:] - Array(H_c[1], inds(H_c[1]))[j,:,:]))
    display(norm(Array(H_c2[2], inds(H_c2[2]))[j,:,:] - Array(H_c[2], inds(H_c[2]))[j,:,:]))
end


println("HElo!")
update_H(H, bc_params, t_list[3])
println("-------------------------")
display(Array(H_c2[1], inds(H_c2[1]))[1,:,:])

let 
    H_man = H_MPO_manual(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
    display(Array(H_man[1], inds(H_man[1]))[1,:,:])
    for i in 2:2
        H_c = piecewise_H_MPO_v2(i, pt, qt, ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
        H_c2 = H_MPO_control(ground_freq, rot_freq, self_kerr, cross_kerr, bc_params, t_list[i], dipole, N ,sites)
        H_man = update_H(H_man, bc_params, t_list[i])
        println("Step $i")
        display(Array(H_c[1], inds(H_c[1]))[1,:,:])
        # display(Array(H_c[1], inds(H_c[1])))
        # display(Array(H_man[1], inds(H_man[1])))
        # println(H_c[2])
        # println("Step $i")
        # println("Control Difference: ", bcarrier2(t_list[i], bc_params, 3) - qt[2,i])
        # println("Difference: ", norm(matrix_form(H_c, sites) - matrix_form(H_c2, sites)))
        # println("Difference: ", norm(H_c - H_man))
    end
end

