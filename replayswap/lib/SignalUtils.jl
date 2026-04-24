module SignalUtils

using Statistics, LinearAlgebra
using SignalAnalysis

export stdize, estimate_mmse_channel, estimate_omp_channel, argminphase

function stdize(y)
    y = samples(y)
    ȳ = mean(y)
    μ = y .- ȳ
    σ = sqrt(mean(abs2, μ))
    return μ ./ σ
end

function estimate_mmse_channel(y_train::Vector, x_train::Vector, L_h::Int; σ²::Float64=1e-3)
    N = length(y_train)
    @assert length(x_train) ≥ N
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    XtX = X' * X
    return (XtX + σ² * I(L_h)) \ (X' * y_train)
end

function estimate_omp_channel(y_train::Vector, x_train::Vector, L_h::Int, k::Int)
    N = length(y_train)
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    residual = copy(y_train)
    support = Int[]
    for _ in 1:k
        correlations = abs.(X' * residual)
        j = argmax(correlations)
        push!(support, j)
        Xs = X[:, support]
        h_tmp = Xs \ y_train
        residual = y_train - Xs * h_tmp
    end
    h_omp = zeros(ComplexF64, L_h)
    h_omp[support] = X[:, support] \ y_train
    return h_omp
end

"Find phase rotation minimizing sign-BER vs reference."
function argminphase(x̂, x)
    bers = Float64[]
    range = 0.0:0.1:360.0
    for θ in range
        x̂a = x̂ .* exp(-im * deg2rad(-θ))
        ber = sum(abs, sign.(real.(x̂a)) .!= sign.(real.(x)))
        push!(bers, ber)
    end
    val, id = findmin(bers)
    return (x̂ .* exp(-im * deg2rad(-range[id])), range[id])
end

end # module
