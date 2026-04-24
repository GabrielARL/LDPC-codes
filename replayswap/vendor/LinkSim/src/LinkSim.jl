module LinkSim

export Code, FEC_create_code, initcode, encode,
       sum_product_decode, get_H_sparse, is_valid_codeword,
       jsdc_qpsk_manual,
       jsdc_bpsk_manual     # <-- add

module CodesLDPC
    include(joinpath(@__DIR__, "Codes", "LDPCCore_trimmed.jl"))
    include(joinpath(@__DIR__, "Codes", "QPSKSpinGrad.jl"))
    include(joinpath(@__DIR__, "Codes", "BPSKSpinGrad.jl"))  # <-- add
end

using .CodesLDPC: Code, FEC_create_code, initcode, encode,
                  sum_product_decode, get_H_sparse, is_valid_codeword,
                  jsdc_qpsk_manual,
                  jsdc_bpsk_manual   # <-- add

end # module LinkSim
