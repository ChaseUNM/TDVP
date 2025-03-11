using ITensors, ITensorMPS

# Example MPO (just a random example for 3 sites)
# Define the dimension for each site
d = 3
N = 3  # Number of sites

# Create a random MPO (just for demonstration)
# Here we use simple 3x3 tensors for each site of the MPO
# Note: You may already have an MPO from your computation
MPO = [randITensor(Index(d), Index(d), Index(d)) for _ in 1:N]

# Function to reconstruct the matrix from an MPO
function reconstruct_matrix(MPO)
    # Start with the first MPO tensor
    result = MPO[1]
    
    # Sequentially contract the tensors in the MPO to form the full matrix
    for i in 2:length(MPO)
        result *= MPO[i]  # Contract current MPO tensor with the result
    end
    
    return result
end

# Reconstruct the matrix from the MPO
reconstructed_matrix = reconstruct_matrix(MPO)

# Display the reconstructed matrix
println(reconstructed_matrix)