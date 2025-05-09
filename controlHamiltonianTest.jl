include("hamiltonian.jl")

N = 2
d = 4
sites = siteinds("Qudit", N, dim = d)

ground_freq = [4.80595, 4.8601]*(2*pi)
rot_freq = sum(ground_freq)/N*ones(N)
dipole = [0 0.005; 0 0]*(2*pi)
self_kerr = [0.2, 0.2]*(2*pi)
cross_kerr = zeros(2, 2)

H_s = H_MPO_v2(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)

pt = npzread("2Qubit_bspline_pt.npy")
qt = npzread("2Qubit_bspline_qt.npy")

H_t = piecewise_H_MPO_v2(1, pt_correct_unit, qt_correct_unit, ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)

H_s1 = H_s[1]
H_s2 = H_s[2]

H_t1 = H_t[1]
H_t2 = H_t[2]

H_s1_arr = Array(H_s1, inds(H_s1))
H_s2_arr = Array(H_s2, inds(H_s2))

H_t1_arr = Array(H_t1, inds(H_t1))
H_t2_arr = Array(H_t2, inds(H_t2))

let 
    for i = 1:4
        println("Bond Dimension: ", i) 
        println("First Site: " )
        display(H_s1_arr[i,:,:])
        display(H_t1_arr[i,:,:])
        println("Second Site: ")
        display(H_s2_arr[i,:,:])
        display(H_t2_arr[i,:,:])
    end
end
