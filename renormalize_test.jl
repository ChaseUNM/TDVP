using ITensorMPS, ITensors 
using LinearAlgebra, Random

Random.seed!(42)
#Create random tensor
i = Index(4)
j = Index(4)
k = Index(4)

M = randomITensor(i, j, k)
M = M/norm(M)
println("Norm of tensor: ", norm(M))
#Perform SVD 
U, S, V = svd(M, i, cutoff = 0.3)
println(S)
new_M = U*S*V
println("Error: ", norm(M - new_M))
println("New Norm: ", norm(new_M))

#Now let's renormalize to see what happens 
S = S/norm(S)
renormalize_newM = U*S*V
println("Renormalized Norm: ", norm(renormalize_newM))
println("New Error: ", norm(M - renormalize_newM))