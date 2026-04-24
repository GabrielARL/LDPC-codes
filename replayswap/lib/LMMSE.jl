module LMMSE

export lmmse_ff, equalize_soft

using LinearAlgebra
using ToeplitzMatrices

function lmmse_ff(h::AbstractVector{<:Complex}, σ2::Real; M::Int=9, D::Int=0)
    L = length(h)
    r = zeros(ComplexF64, M)
    @inbounds for k in 0:M-1
        s = 0.0 + 0.0im
        for i in 1:L
            j = i + k
            if 1 ≤ j ≤ L
                s += h[i]*conj(h[j])
            end
        end
        r[k+1] = s
    end
    r[1] += σ2
    R = Toeplitz(r, conj(r))

    p = zeros(ComplexF64, M)
    @inbounds for m in 0:M-1
        idx = D + 1 + m
        if 1 ≤ idx ≤ L
            p[m+1] = conj(h[idx])
        end
    end

    w = R \ p
    return w, D
end

function equalize_soft(y_use::Vector{ComplexF64}, w::Vector{ComplexF64})
    M = length(w)
    (length(y_use) < M) && return ComplexF64[]
    y_eq = Vector{ComplexF64}(undef, length(y_use) - M + 1)
    @inbounds for n in 1:length(y_eq)
        y_eq[n] = sum(w .* @view y_use[n:n+M-1])
    end
    return y_eq
end

end # module

