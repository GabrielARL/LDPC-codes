module FECBasic

using ..FEC

export init_systematic_code,
       encode_systematic,
       is_valid,
       bp_decode_spa,
       decode_soft,
       bpsk_hard_bits_on_data,
       bpsk_llrs_on_data,
       qpsk_hard_bits_on_data,
       qpsk_llrs_on_data,
       psk8_hard_bits_on_data,
       psk8_llrs_on_data,
       hard_bits_on_data,
       llrs_on_data

const init_systematic_code   = FEC.init_systematic_code
const encode_systematic      = FEC.encode_systematic
const is_valid               = FEC.is_valid
const bp_decode_spa          = FEC.bp_decode_spa
const decode_soft            = FEC.decode_soft
const bpsk_hard_bits_on_data = FEC.bpsk_hard_bits_on_data
const bpsk_llrs_on_data      = FEC.bpsk_llrs_on_data
const qpsk_hard_bits_on_data = FEC.qpsk_hard_bits_on_data
const qpsk_llrs_on_data      = FEC.qpsk_llrs_on_data
const psk8_hard_bits_on_data = FEC.psk8_hard_bits_on_data
const psk8_llrs_on_data      = FEC.psk8_llrs_on_data
const hard_bits_on_data      = FEC.hard_bits_on_data
const llrs_on_data           = FEC.llrs_on_data

end # module FECBasic
