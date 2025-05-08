using ITensors

#Converts arr to integer
function bitarr_to_int_p_1(arr)
    return sum(arr .* (2 .^ collect(length(arr)-1:-1:0))) + 1
end

#Converts integer to bit array
function int_to_bitarr(num, last_n)
    num1 = num - 1
    bit_str = bitstring(UInt16(num1))
    bit_arr = split(bit_str, "")
    bit_arr = parse.(Int64, bit_arr)
    bit_arr = last(bit_arr, last_n)
    return bit_arr
end

#Reconstructs an array from an MPS. 
#N is the number of qubits, d is the number of energy levels per system (d = 2 when working with qubits)
#psi is the given MPS, sites indices are the site of the MPS, for sites you can always input 'siteinds(psi)'
function reconstruct_arr(psi)
    sites = siteinds(psi)
    N = length(psi)
    total_length = dim(sites)
    reconstruct_arr = zeros(ComplexF64, total_length)
    for i = 1:total_length
        el = int_to_bitarr(i, N)
        println("el: $el")
        el .+= 1
        V = ITensor(1.)
        for j = 1:N
            # println("V: ", psi[j]*state(s[j], el[j]))
            V *= (psi[j]*state(sites[j], el[j]))
        end
        # println("V: ", V)
        v = scalar(V)
        loc = bitarr_to_int_p_1(reverse(el .- 1))
        println(loc)
        reconstruct_arr[loc]  = v       
    end
    return reconstruct_arr
end

function reconstruct_arr_v2(psi)
    M = psi[1]
    N = length(psi)
    for i = 2:N 
        M *= psi[i]
    end
    M_c = contract(M)
    M_arr = Array(M_c, inds(M_c))
    vec_length = prod(dim(inds(M_c)))
    M_vec = reshape(M_arr, vec_length)
    return M_vec 
end

# N = 2 
# M = MPS(N)
# M1 = [0 0 sqrt(2); 0 sqrt(2) 0]
# M2 = [0 0; 1 0; 0 1]
# i1 = Index(2, "Link")
# i2 = Index(3, "Site")
# i3 = Index(3, "Site")
# M1_t = ITensor(M1, [i1, i2])
# M2_t = ITensor(M2, [i3, i1])
# M[1] = M1_t 
# M[2] = M2_t
# println(M[1])
# println(M[2])
# M_contract = contract(M[1]*M[2])
# M_arr = Array(M_contract, inds(M_contract))
# println(M_contract)
# println(M_arr)
# @time begin
# println(reconstruct_arr(3, N, M, siteinds(M)))
# end

# @time begin
# println(reconstruct_arr_v2(M))
# end