using LinearAlgebra, Plots

function At(t)
    A = zeros(2,2)
    A[1,1] = -t 
    A[2,2] = -t 
    return A 
end

function At2(t)
    A = zeros(2,2)
    A[1,1] = -t 
    A[1,2] = 1 
    A[2,2] = -t
    return A 
end

function At3(t)
    A = zeros(2,2)
    A[1,1] = -cos(t) 
    A[2,2] = cos(t)
    return A 
end

function At4(t)
    A = zeros(2,2)
    A[1,1] = -cos(t) 
    A[1,2] = 1 
    A[2,2] = -cos(t)
    return A 
end


function Bt(t)
    B = zeros(2,2)
    B[1,1] = cos(t)
    B[1,2] = sin(t)
    B[2,1] = sin(t)
    B[2,2] = -cos(t) 
    return B 
end

function Bt2(t)
    B = zeros(2,2)
    B[1,1] = cos(t)
    B[1,2] = -1
    B[2,1] = 1
    B[2,2] = -cos(t) 
    return B 
end
function Ct(t)
    C = At3(t) + Bt2(t)
    return C 
end

function IMR(A, y, h)
    LHS = I - h/2*A 
    RHS = (I + h/2*A)*y 
    y_n = LHS\RHS
    return y_n 
end

#Need to create initial condition, and we're just doing 1 time-step
let 
    n = 100
    err = zeros(n)
    count = 1
    finalN = 5
    true_sol_err = zeros(n)
    true_sol_err2 = zeros(n)
    for T in  10.0 .^-LinRange(1,finalN,n)
        y0 = [1,1]
        t0 = 0.0 
        nsteps = 1
        h = (T - t0)/nsteps 

        y_sol = IMR(Ct(t0 + h/2), y0, h)

        y_sol_1 = IMR(At3(t0 + h/2), y0, h)

        y_sol_2 = IMR(Bt(t0 + h/2), y_sol_1, h)

        true_sol = zeros(2)
        true_sol[1] = cos(T) - sin(T)
        true_sol[2] = sin(T) + cos(T)
        true_sol_err[count] = norm(true_sol - y_sol)
        true_sol_err2[count] = norm(true_sol - y_sol_2)

        println("Error for h = $h: ", norm(y_sol - y_sol_2))
        err[count] = norm(y_sol - y_sol_2)
        count += 1
    end
    # C = err[1]/(10^-)
    plot(10.0 .^-LinRange(1,finalN,n), err, label = "Error", xscale =:log10,yscale =:log10, legend=:topleft)
    plot!(10.0 .^-LinRange(1,finalN,n), (10 .^-LinRange(1,finalN,n)).^3, linestyle=:dash, label = "Comparison", xscale =:log10, yscale =:log10)
    plot!(10.0 .^-LinRange(1,finalN,n), true_sol_err, linestyle=:dash, label = "Error between true solution and non-splitting method", xscale =:log10, yscale =:log10)
    plot!(10.0 .^-LinRange(1,finalN,n), true_sol_err2, linestyle=:dash, label = "Error between true solution and splitting method", xscale =:log10, yscale =:log10)
end


