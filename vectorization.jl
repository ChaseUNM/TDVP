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
function reconstruct_arr(d, N, psi, sites)
    reconstruct_arr = zeros(ComplexF64, d^N)
    for i = 1:d^N
        el = reverse(int_to_bitarr(i, N))
        el .+= 1
        V = ITensor(1.)
        for j = 1:N
            # println("V: ", psi[j]*state(s[j], el[j]))
            V *= (psi[j]*state(sites[j], el[j]))
        end
        # println("V: ", V)
        v = scalar(V)
        loc = bitarr_to_int_p_1(reverse(el .- 1))
        reconstruct_arr[loc]  = v       
    end
    return reconstruct_arr
end