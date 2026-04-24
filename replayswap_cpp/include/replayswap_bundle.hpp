#ifndef REPLAYSWAP_CPP_REPLAYSWAP_BUNDLE_HPP
#define REPLAYSWAP_CPP_REPLAYSWAP_BUNDLE_HPP

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace replayswap_cpp {

using cd = std::complex<double>;

struct ComplexMatrix {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<cd> data;

    cd& operator()(std::int64_t row, std::int64_t col) {
        return data[static_cast<std::size_t>(row * cols + col)];
    }

    const cd& operator()(std::int64_t row, std::int64_t col) const {
        return data[static_cast<std::size_t>(row * cols + col)];
    }
};

struct IntMatrix {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<int> data;

    int& operator()(std::int64_t row, std::int64_t col) {
        return data[static_cast<std::size_t>(row * cols + col)];
    }

    const int& operator()(std::int64_t row, std::int64_t col) const {
        return data[static_cast<std::size_t>(row * cols + col)];
    }
};

struct BundleManifest {
    std::string root_dir;
    std::string bundle_name;
    std::string ldpc_h_file;

    std::string raw_packets_file;
    std::string raw_frame_ids_file;
    std::string raw_block_ids_file;
    std::string raw_bestd_file;
    std::string raw_cw_true_file;

    std::string rsc_y_file;
    std::string rsc_u64_file;
    std::string rsc_b128_file;
    std::string rsc_h_file;
    std::string rsc_corr_file;
    std::string rsc_frame_ids_file;
    std::string rsc_default_order_file;

    std::vector<double> pilot_fracs;

    int raw_start_frame = 1;
    int raw_n_per_p = 20;
    int raw_h_len = 20;
    double raw_rho_ls = 1e-2;
    double raw_lambda = 2.0;
    double raw_lambda_pil = 20.0;
    double raw_gamma = 1e-3;
    double raw_eta = 1.0;
    int raw_k_sparse = 4;
    int raw_max_iter_opt = 20;

    double rsc_corr_thr = 0.10;
    int rsc_nblk = 200;
    int rsc_seed_sel = 12648430;
    int rsc_start = 1;
    int rsc_turbo_iters = 2;
    double rsc_sigma2_init = 1.30;
    int rsc_eq_sigma2_iters = 1;
    double rsc_llr_clip = 25.0;
    double rsc_default_order_corr_thr = 0.10;
    int rsc_default_order_seed = 12648430;
};

struct BundleData {
    BundleManifest manifest;

    ComplexMatrix raw_packets;
    IntMatrix raw_cw_true;
    std::vector<int> raw_frame_ids;
    std::vector<int> raw_block_ids;
    std::vector<int> raw_bestd;

    ComplexMatrix rsc_y;
    IntMatrix rsc_u64;
    IntMatrix rsc_b128;
    ComplexMatrix rsc_h;
    std::vector<double> rsc_corr;
    std::vector<int> rsc_frame_ids;
    std::vector<int> rsc_default_order;
};

BundleManifest read_manifest(const std::string& path);
BundleData load_bundle(const std::string& manifest_or_dir);

std::string join_path(const std::string& a, const std::string& b);
std::vector<cd> row_as_complex(const ComplexMatrix& mat, int row, int max_cols = -1);
std::vector<int> row_as_int(const IntMatrix& mat, int row, int max_cols = -1);

}  // namespace replayswap_cpp

#endif
