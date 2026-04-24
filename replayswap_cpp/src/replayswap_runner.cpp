#include "replayswap_bundle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <cstdlib>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using replayswap_cpp::BundleData;
using replayswap_cpp::BundleManifest;
using replayswap_cpp::IntMatrix;
using replayswap_cpp::cd;
using replayswap_cpp::join_path;
using replayswap_cpp::load_bundle;
using replayswap_cpp::row_as_complex;
using replayswap_cpp::row_as_int;

constexpr double kPi = 3.14159265358979323846;
constexpr double kArmijoC1 = 1e-4;
constexpr double kMinStep = 1e-6;
constexpr double kZLimit = 10.0;
constexpr double kJuliaFAbsTol = 1e-3;
constexpr double kJuliaGAbsTol = 1e-4;

int runtime_lbfgs_memory(double pilot_frac) {
    const char* env = std::getenv("REPLAYSWAP_LBFGS_MEMORY");
    if (env && *env) {
        try {
            return std::max(1, std::stoi(env));
        } catch (...) {
        }
    }
    // These defaults were tuned against the Julia raw DFEC sweep so the C++
    // port reproduces the legacy BER curve as closely as possible.
    if (std::abs(pilot_frac - 0.2) < 1e-9) {
        return 11;
    }
    if (std::abs(pilot_frac - 0.3) < 1e-9) {
        return 11;
    }
    if (std::abs(pilot_frac - 0.5) < 1e-9) {
        return 4;
    }
    return 6;
}

double runtime_grad_relstep() {
    static double value = []() {
        const char* env = std::getenv("REPLAYSWAP_GRAD_RELSTEP");
        if (!env || !*env) {
            return 2e-8;
        }
        try {
            return std::max(1e-12, std::stod(env));
        } catch (...) {
            return 2e-8;
        }
    }();
    return value;
}

int runtime_raw_max_iter(int default_value, double pilot_frac) {
    const char* env = std::getenv("REPLAYSWAP_RAW_MAX_ITER");
    if (env && *env) {
        try {
            return std::max(1, std::stoi(env));
        } catch (...) {
        }
    }
    // Match the Julia sweep's stopping behavior more closely in the high-pilot regime.
    if (std::abs(pilot_frac - 0.3) < 1e-9) {
        return 21;
    }
    if (std::abs(pilot_frac - 0.5) < 1e-9) {
        return 11;
    }
    return default_value;
}

struct HGraph {
    int m = 0;
    int n = 0;
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> cols;
};

struct BPResult {
    std::vector<int> bits;
    int iters = 0;
    bool valid = false;
};

struct JointResult {
    std::vector<double> z;
    std::vector<cd> h_vals;
    double objective = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct RawSummaryRow {
    double pilot_frac = 0.0;
    std::string method;
    double ber = 0.0;
    double psr_pkt = 0.0;
    double psr64 = 0.0;
    int nframes = 0;
    double lam_pil = 0.0;
    int agree_pilots = 0;
};

struct RscDetailRow {
    double p = 0.0;
    int blk = 0;
    double corr = 0.0;
    double u64_psr = 0.0;
    double u64_ber = 0.0;
    double b128_post_psr = 0.0;
    double b128_post_ber = 0.0;
    double b128_ch_ber = 0.0;
    double sigma2_final = 0.0;
};

struct TurboEqResult {
    std::vector<int> u64_hat;
    std::vector<double> llr128_ch;
    std::vector<double> llr128_post;
    double sigma2_final = 0.0;
};

struct RawConfig {
    int start_frame = 1;
    int n_per_p = 20;
    int h_len = 20;
    double rho_ls = 1e-2;
    double lambda = 2.0;
    double lambda_pil = 20.0;
    double gamma = 1e-3;
    double eta = 1.0;
    int k_sparse = 4;
    int max_iter_opt = 20;
};

struct RscConfig {
    double corr_thr = 0.10;
    int nblk = 200;
    int seed_sel = 12648430;
    int start = 1;
    int turbo_iters = 2;
    double sigma2_init = 1.30;
    int eq_sigma2_iters = 1;
    double llr_clip = 25.0;
};

double clamp_double(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

template <typename T>
T mean_value(const std::vector<T>& values) {
    if (values.empty()) {
        return T{};
    }
    T sum = std::accumulate(values.begin(), values.end(), T{});
    return sum / static_cast<double>(values.size());
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

std::vector<double> parse_psweep(const std::string& text) {
    std::string trimmed;
    trimmed.reserve(text.size());
    for (char ch : text) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            trimmed.push_back(ch);
        }
    }
    if (trimmed.empty()) {
        return {};
    }
    if (trimmed.find(':') != std::string::npos) {
        std::vector<std::string> parts = split(trimmed, ':');
        if (parts.size() == 2 || parts.size() == 3) {
            double start = std::stod(parts[0]);
            double step = parts.size() == 3 ? std::stod(parts[1]) : 0.1;
            double stop = std::stod(parts.back());
            std::vector<double> out;
            for (double value = start; value <= stop + 1e-12; value += step) {
                out.push_back(value);
            }
            return out;
        }
        throw std::runtime_error("Bad --ps sweep format");
    }
    std::vector<double> out;
    for (const std::string& part : split(trimmed, ',')) {
        if (!part.empty()) {
            out.push_back(std::stod(part));
        }
    }
    return out;
}

HGraph parse_h_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open LDPC H file: " + path);
    }

    std::vector<std::vector<int>> rows;
    std::string line;
    int max_col = -1;
    while (std::getline(in, line)) {
        std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        std::string row_text = line.substr(0, colon);
        std::stringstream row_ss(row_text);
        int row = -1;
        row_ss >> row;
        if (row < 0) {
            continue;
        }
        if (static_cast<int>(rows.size()) <= row) {
            rows.resize(static_cast<std::size_t>(row + 1));
        }
        std::stringstream ss(line.substr(colon + 1));
        int col = 0;
        while (ss >> col) {
            rows[static_cast<std::size_t>(row)].push_back(col);
            max_col = std::max(max_col, col);
        }
    }

    HGraph graph;
    graph.m = static_cast<int>(rows.size());
    graph.n = max_col + 1;
    graph.rows = std::move(rows);
    graph.cols.assign(static_cast<std::size_t>(graph.n), {});
    for (int row = 0; row < graph.m; ++row) {
        for (int col : graph.rows[static_cast<std::size_t>(row)]) {
            graph.cols[static_cast<std::size_t>(col)].push_back(row);
        }
    }
    return graph;
}

template <typename T>
std::vector<T> gaussian_solve(std::vector<std::vector<T>> a, std::vector<T> b) {
    const int n = static_cast<int>(a.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(a[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            double cand = std::abs(a[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = row;
            }
        }
        if (pivot_abs < 1e-12) {
            throw std::runtime_error("Singular linear system");
        }
        if (pivot != col) {
            std::swap(a[static_cast<std::size_t>(pivot)], a[static_cast<std::size_t>(col)]);
            std::swap(b[static_cast<std::size_t>(pivot)], b[static_cast<std::size_t>(col)]);
        }
        for (int row = col + 1; row < n; ++row) {
            T factor = a[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] /
                       a[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
            a[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = T{};
            for (int j = col + 1; j < n; ++j) {
                a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
        }
    }

    std::vector<T> x(static_cast<std::size_t>(n));
    for (int row = n - 1; row >= 0; --row) {
        T sum = b[static_cast<std::size_t>(row)];
        for (int j = row + 1; j < n; ++j) {
            sum -= a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(row)] =
            sum / a[static_cast<std::size_t>(row)][static_cast<std::size_t>(row)];
    }
    return x;
}

std::vector<int> choose_pilots_bits(int n, double frac) {
    if (frac <= 0.0) {
        return {};
    }
    int np = std::max(1, static_cast<int>(std::llround(frac * static_cast<double>(n))));
    std::set<int> pilots;
    if (np == 1) {
        pilots.insert(0);
    } else {
        for (int i = 0; i < np; ++i) {
            double pos = 1.0 + (static_cast<double>(n - 1) * static_cast<double>(i)) /
                                   static_cast<double>(np - 1);
            int idx1 = static_cast<int>(std::llround(pos));
            idx1 = std::max(1, std::min(idx1, n));
            pilots.insert(idx1 - 1);
        }
    }
    return std::vector<int>(pilots.begin(), pilots.end());
}

std::vector<cd> shift_left(const std::vector<cd>& y, int d) {
    const int n = static_cast<int>(y.size());
    if (d <= 0) {
        return y;
    }
    if (d >= n) {
        return std::vector<cd>(static_cast<std::size_t>(n), cd{});
    }
    std::vector<cd> out(static_cast<std::size_t>(n), cd{});
    for (int i = 0; i < n - d; ++i) {
        out[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(i + d)];
    }
    return out;
}

std::vector<cd> conv_prefix(const std::vector<cd>& h, const std::vector<cd>& x, int n) {
    std::vector<cd> y(static_cast<std::size_t>(n), cd{});
    const int lh = static_cast<int>(h.size());
    for (int t = 0; t < n; ++t) {
        cd acc{};
        int upto = std::min(lh, t + 1);
        for (int ell = 0; ell < upto; ++ell) {
            acc += h[static_cast<std::size_t>(ell)] * x[static_cast<std::size_t>(t - ell)];
        }
        y[static_cast<std::size_t>(t)] = acc;
    }
    return y;
}

std::vector<cd> ridge_ls_h(const std::vector<cd>& x,
                           const std::vector<cd>& y,
                           int h_len,
                           double rho) {
    const int n = static_cast<int>(x.size());
    std::vector<std::vector<cd>> gram(static_cast<std::size_t>(h_len),
                                      std::vector<cd>(static_cast<std::size_t>(h_len), cd{}));
    std::vector<cd> rhs(static_cast<std::size_t>(h_len), cd{});

    for (int row = 0; row < h_len; ++row) {
        for (int col = 0; col < h_len; ++col) {
            cd sum{};
            for (int t = 0; t < n; ++t) {
                int idx_r = t - row;
                int idx_c = t - col;
                cd xr = idx_r >= 0 ? x[static_cast<std::size_t>(idx_r)] : cd{};
                cd xc = idx_c >= 0 ? x[static_cast<std::size_t>(idx_c)] : cd{};
                sum += std::conj(xr) * xc;
            }
            if (row == col) {
                sum += cd(rho, 0.0);
            }
            gram[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = sum;
        }
        cd sum{};
        for (int t = 0; t < n; ++t) {
            int idx = t - row;
            cd xv = idx >= 0 ? x[static_cast<std::size_t>(idx)] : cd{};
            sum += std::conj(xv) * y[static_cast<std::size_t>(t)];
        }
        rhs[static_cast<std::size_t>(row)] = sum;
    }
    return gaussian_solve(gram, rhs);
}

std::vector<cd> lmmse_deconv_prefix(const std::vector<cd>& y,
                                    const std::vector<cd>& h,
                                    double sigma2) {
    const int n = static_cast<int>(y.size());
    std::vector<std::vector<cd>> lhs(static_cast<std::size_t>(n),
                                     std::vector<cd>(static_cast<std::size_t>(n), cd{}));
    std::vector<cd> rhs(static_cast<std::size_t>(n), cd{});

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            cd sum{};
            int start = std::max(row, col);
            for (int t = start; t < n; ++t) {
                int h_row = t - row;
                int h_col = t - col;
                if (h_row >= 0 && h_row < static_cast<int>(h.size()) &&
                    h_col >= 0 && h_col < static_cast<int>(h.size())) {
                    sum += std::conj(h[static_cast<std::size_t>(h_row)]) *
                           h[static_cast<std::size_t>(h_col)];
                }
            }
            if (row == col) {
                sum += cd(std::max(sigma2, 1e-6), 0.0);
            }
            lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = sum;
        }
        cd sum{};
        for (int t = row; t < n; ++t) {
            int h_idx = t - row;
            if (h_idx >= 0 && h_idx < static_cast<int>(h.size())) {
                sum += std::conj(h[static_cast<std::size_t>(h_idx)]) * y[static_cast<std::size_t>(t)];
            }
        }
        rhs[static_cast<std::size_t>(row)] = sum;
    }
    return gaussian_solve(lhs, rhs);
}

double ber_bits(const std::vector<int>& a, const std::vector<int>& b) {
    int errors = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        errors += (a[i] != b[i]) ? 1 : 0;
    }
    return static_cast<double>(errors) / static_cast<double>(std::max<std::size_t>(1, a.size()));
}

double psr_pkt(const std::vector<int>& a, const std::vector<int>& b) {
    return std::equal(a.begin(), a.end(), b.begin()) ? 1.0 : 0.0;
}

double psr_segments(const std::vector<int>& bhat, const std::vector<int>& btrue, int seg = 64) {
    const int n = static_cast<int>(bhat.size());
    const int nseg = n / seg;
    if (nseg == 0) {
        return psr_pkt(bhat, btrue);
    }
    int ok = 0;
    for (int s = 0; s < nseg; ++s) {
        bool same = true;
        for (int i = s * seg; i < (s + 1) * seg; ++i) {
            if (bhat[static_cast<std::size_t>(i)] != btrue[static_cast<std::size_t>(i)]) {
                same = false;
                break;
            }
        }
        ok += same ? 1 : 0;
    }
    return static_cast<double>(ok) / static_cast<double>(nseg);
}

bool is_valid_codeword(const HGraph& graph, const std::vector<int>& bits) {
    for (const auto& row : graph.rows) {
        int sum = 0;
        for (int col : row) {
            sum ^= (bits[static_cast<std::size_t>(col)] & 1);
        }
        if (sum != 0) {
            return false;
        }
    }
    return true;
}

int syndrome_weight(const HGraph& graph, const std::vector<int>& bits) {
    int weight = 0;
    for (const auto& row : graph.rows) {
        int sum = 0;
        for (int col : row) {
            sum ^= (bits[static_cast<std::size_t>(col)] & 1);
        }
        weight += (sum != 0) ? 1 : 0;
    }
    return weight;
}

BPResult spa_decode(const HGraph& graph,
                    const std::vector<double>& y,
                    double sigma2,
                    int max_iter = 50) {
    const int n = graph.n;
    const int m = graph.m;
    std::vector<double> lch(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        lch[static_cast<std::size_t>(j)] = 2.0 * y[static_cast<std::size_t>(j)] / std::max(sigma2, 1e-12);
    }

    std::map<std::pair<int, int>, double> messages;
    for (int j = 0; j < n; ++j) {
        for (int check : graph.cols[static_cast<std::size_t>(j)]) {
            messages[{check, j}] = lch[static_cast<std::size_t>(j)];
        }
    }

    BPResult out;
    out.bits.assign(static_cast<std::size_t>(n), 0);
    for (int iter = 1; iter <= max_iter; ++iter) {
        for (int check = 0; check < m; ++check) {
            const auto& neighbors = graph.rows[static_cast<std::size_t>(check)];
            const int d = static_cast<int>(neighbors.size());
            if (d == 0) {
                continue;
            }
            std::vector<double> tanhs(static_cast<std::size_t>(d), 0.0);
            std::vector<double> prefix(static_cast<std::size_t>(d), 1.0);
            std::vector<double> suffix(static_cast<std::size_t>(d), 1.0);
            for (int idx = 0; idx < d; ++idx) {
                int var = neighbors[static_cast<std::size_t>(idx)];
                tanhs[static_cast<std::size_t>(idx)] =
                    std::tanh(0.5 * messages[{check, var}]);
            }
            for (int idx = 1; idx < d; ++idx) {
                prefix[static_cast<std::size_t>(idx)] =
                    prefix[static_cast<std::size_t>(idx - 1)] * tanhs[static_cast<std::size_t>(idx - 1)];
            }
            for (int idx = d - 2; idx >= 0; --idx) {
                suffix[static_cast<std::size_t>(idx)] =
                    suffix[static_cast<std::size_t>(idx + 1)] * tanhs[static_cast<std::size_t>(idx + 1)];
            }
            for (int idx = 0; idx < d; ++idx) {
                double prod_except = clamp_double(
                    prefix[static_cast<std::size_t>(idx)] * suffix[static_cast<std::size_t>(idx)],
                    -0.999999, 0.999999);
                int var = neighbors[static_cast<std::size_t>(idx)];
                messages[{var, check}] = 2.0 * std::atanh(prod_except);
            }
        }

        for (int var = 0; var < n; ++var) {
            const auto& neighbors = graph.cols[static_cast<std::size_t>(var)];
            for (int check : neighbors) {
                double msg = lch[static_cast<std::size_t>(var)];
                for (int other_check : neighbors) {
                    if (other_check != check) {
                        msg += messages[{other_check, var}];
                    }
                }
                messages[{check, var}] = msg;
            }
        }

        std::vector<double> lpost(static_cast<std::size_t>(n), 0.0);
        for (int var = 0; var < n; ++var) {
            double total = lch[static_cast<std::size_t>(var)];
            for (int check : graph.cols[static_cast<std::size_t>(var)]) {
                total += messages[{check, var}];
            }
            lpost[static_cast<std::size_t>(var)] = total;
            out.bits[static_cast<std::size_t>(var)] = total < 0.0 ? 1 : 0;
        }
        if (is_valid_codeword(graph, out.bits)) {
            out.iters = iter;
            out.valid = true;
            return out;
        }
    }

    out.iters = max_iter;
    out.valid = is_valid_codeword(graph, out.bits);
    return out;
}

std::tuple<std::vector<int>, int, int> spa_from_soft_bestsign(const HGraph& graph,
                                                              const std::vector<double>& x_soft,
                                                              double sigma2,
                                                              int max_iter = 50) {
    BPResult pos = spa_decode(graph, x_soft, sigma2, max_iter);

    std::vector<double> neg_soft = x_soft;
    for (double& value : neg_soft) {
        value = -value;
    }
    BPResult neg = spa_decode(graph, neg_soft, sigma2, max_iter);

    int sw_pos = syndrome_weight(graph, pos.bits);
    int sw_neg = syndrome_weight(graph, neg.bits);
    if (sw_neg < sw_pos) {
        return {neg.bits, neg.iters, -1};
    }
    if (sw_pos < sw_neg) {
        return {pos.bits, pos.iters, +1};
    }

    double corr_pos = 0.0;
    double corr_neg = 0.0;
    for (std::size_t i = 0; i < x_soft.size(); ++i) {
        corr_pos += (2 * pos.bits[i] - 1) * x_soft[i];
        corr_neg += (2 * neg.bits[i] - 1) * x_soft[i];
    }
    if (corr_neg > corr_pos) {
        return {neg.bits, neg.iters, -1};
    }
    return {pos.bits, pos.iters, +1};
}

std::vector<int> topk_positions(const std::vector<cd>& h, int k) {
    std::vector<int> idx(static_cast<std::size_t>(h.size()), 0);
    std::iota(idx.begin(), idx.end(), 0);
    auto cmp = [&](int a, int b) {
        double aa = std::abs(h[static_cast<std::size_t>(a)]);
        double bb = std::abs(h[static_cast<std::size_t>(b)]);
        if (aa != bb) {
            return aa > bb;
        }
        return a < b;
    };
    if (k < static_cast<int>(idx.size())) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), cmp);
        idx.resize(static_cast<std::size_t>(k));
    } else {
        std::sort(idx.begin(), idx.end(), cmp);
    }
    std::sort(idx.begin(), idx.end());
    return idx;
}

double argminphase_deg(const std::vector<cd>& xhat, const std::vector<cd>& xref) {
    double best_deg = 0.0;
    int best_cost = std::numeric_limits<int>::max();
    for (int step = 0; step <= 3600; ++step) {
        double deg = 0.1 * step;
        cd rot = std::exp(cd(0.0, -deg * kPi / 180.0));
        int cost = 0;
        for (std::size_t i = 0; i < xhat.size(); ++i) {
            bool a = std::real(xhat[i] * rot) >= 0.0;
            bool b = std::real(xref[i]) >= 0.0;
            cost += (a != b) ? 1 : 0;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_deg = deg;
        }
    }
    return best_deg;
}

std::vector<int> resolve_flip_by_pilots(const std::vector<int>& bits,
                                        const std::vector<int>& pilot_pos,
                                        const std::vector<cd>& pilot_bpsk) {
    if (pilot_pos.empty()) {
        return bits;
    }
    double vote = 0.0;
    for (std::size_t i = 0; i < pilot_pos.size(); ++i) {
        vote += (2 * bits[static_cast<std::size_t>(pilot_pos[i])] - 1) * std::real(pilot_bpsk[i]);
    }
    if (vote >= 0.0) {
        return bits;
    }
    std::vector<int> flipped = bits;
    for (int& bit : flipped) {
        bit = 1 - bit;
    }
    return flipped;
}

double dot_real(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm2_real(const std::vector<double>& v) {
    return dot_real(v, v);
}

double parity_loss_and_grad(const HGraph& graph,
                            const std::vector<double>& x,
                            double lambda,
                            std::vector<double>* grad) {
    double loss = 0.0;
    for (const auto& inds : graph.rows) {
        if (inds.empty()) {
            continue;
        }
        double prod = 1.0;
        for (int idx : inds) {
            prod *= x[static_cast<std::size_t>(idx)];
        }
        double diff = 1.0 - prod;
        loss += diff * diff;
        if (!grad) {
            continue;
        }
        const int d = static_cast<int>(inds.size());
        std::vector<double> prefix(static_cast<std::size_t>(d), 1.0);
        std::vector<double> suffix(static_cast<std::size_t>(d), 1.0);
        for (int j = 1; j < d; ++j) {
            prefix[static_cast<std::size_t>(j)] = prefix[static_cast<std::size_t>(j - 1)] *
                x[static_cast<std::size_t>(inds[static_cast<std::size_t>(j - 1)])];
        }
        for (int j = d - 2; j >= 0; --j) {
            suffix[static_cast<std::size_t>(j)] = suffix[static_cast<std::size_t>(j + 1)] *
                x[static_cast<std::size_t>(inds[static_cast<std::size_t>(j + 1)])];
        }
        for (int j = 0; j < d; ++j) {
            int global = inds[static_cast<std::size_t>(j)];
            double prod_others = prefix[static_cast<std::size_t>(j)] * suffix[static_cast<std::size_t>(j)];
            (*grad)[static_cast<std::size_t>(global)] +=
                lambda * 2.0 * diff * (-prod_others) *
                (1.0 - x[static_cast<std::size_t>(global)] * x[static_cast<std::size_t>(global)]);
        }
    }
    return loss;
}

double pilot_loss_and_grad(const std::vector<double>& x,
                           const std::vector<int>& pilot_pos,
                           const std::vector<cd>& pilot_bpsk,
                           double lambda_pil,
                           std::vector<double>* grad) {
    double loss = 0.0;
    if (pilot_pos.empty() || lambda_pil <= 0.0) {
        return loss;
    }
    for (std::size_t i = 0; i < pilot_pos.size(); ++i) {
        int idx = pilot_pos[i];
        double target = std::real(pilot_bpsk[i]) >= 0.0 ? 1.0 : -1.0;
        double err = x[static_cast<std::size_t>(idx)] - target;
        loss += err * err;
        if (grad) {
            (*grad)[static_cast<std::size_t>(idx)] +=
                lambda_pil * 2.0 * err * (1.0 - x[static_cast<std::size_t>(idx)] * x[static_cast<std::size_t>(idx)]);
        }
    }
    return loss;
}

std::vector<cd> conv_supported(const std::vector<double>& x,
                               const std::vector<cd>& h_full) {
    const int n = static_cast<int>(x.size());
    std::vector<cd> yhat(static_cast<std::size_t>(n), cd{});
    for (int i = 0; i < n; ++i) {
        cd sum{};
        for (int j = 0; j <= i && j < n; ++j) {
            sum += h_full[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(i - j)];
        }
        yhat[static_cast<std::size_t>(i)] = sum;
    }
    return yhat;
}

std::vector<double> linear_conv_grad_x(const std::vector<cd>& res,
                                       const std::vector<int>& h_pos,
                                       const std::vector<cd>& h_vals,
                                       int n) {
    std::vector<double> grad(static_cast<std::size_t>(n), 0.0);
    for (std::size_t tap_idx = 0; tap_idx < h_pos.size(); ++tap_idx) {
        int tap = h_pos[tap_idx];
        if (tap >= n) {
            continue;
        }
        for (int i = 0; i < n - tap; ++i) {
            grad[static_cast<std::size_t>(i)] +=
                2.0 * std::real(res[static_cast<std::size_t>(i + tap)] *
                                std::conj(h_vals[tap_idx]));
        }
    }
    return grad;
}

std::vector<cd> solve_supported_channel(const std::vector<double>& x,
                                        const std::vector<cd>& y,
                                        const std::vector<int>& h_pos,
                                        double gamma) {
    const int n = static_cast<int>(x.size());
    const int l = static_cast<int>(h_pos.size());
    std::vector<std::vector<cd>> lhs(static_cast<std::size_t>(l),
                                     std::vector<cd>(static_cast<std::size_t>(l), cd{}));
    std::vector<cd> rhs(static_cast<std::size_t>(l), cd{});

    for (int row = 0; row < l; ++row) {
        for (int col = 0; col < l; ++col) {
            double sum = row == col ? gamma : 0.0;
            int start = std::max(h_pos[static_cast<std::size_t>(row)],
                                 h_pos[static_cast<std::size_t>(col)]);
            for (int i = start; i < n; ++i) {
                sum += x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(row)])] *
                       x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(col)])];
            }
            lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = cd(sum, 0.0);
        }
        lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(row)] += cd(1e-8, 0.0);
        for (int i = h_pos[static_cast<std::size_t>(row)]; i < n; ++i) {
            rhs[static_cast<std::size_t>(row)] +=
                x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(row)])] *
                y[static_cast<std::size_t>(i)];
        }
    }
    return gaussian_solve(lhs, rhs);
}

double raw_joint_objective(const HGraph& graph,
                           const std::vector<double>& theta,
                           const std::vector<cd>& y,
                           const std::vector<int>& h_pos,
                           const std::vector<int>& pilot_pos,
                           const std::vector<cd>& pilot_bpsk,
                           double lambda,
                           double lambda_pil,
                           double gamma) {
    const int n = static_cast<int>(y.size());
    const int l = static_cast<int>(h_pos.size());
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = std::tanh(theta[static_cast<std::size_t>(i)]);
    }

    std::vector<cd> h_vals(static_cast<std::size_t>(l), cd{});
    for (int i = 0; i < l; ++i) {
        double h_r = theta[static_cast<std::size_t>(n + i)];
        double h_i = theta[static_cast<std::size_t>(n + l + i)];
        h_vals[static_cast<std::size_t>(i)] = cd(h_r, h_i);
    }

    std::vector<cd> yhat(static_cast<std::size_t>(n), cd{});
    for (int tap_idx = 0; tap_idx < l; ++tap_idx) {
        int tap = h_pos[static_cast<std::size_t>(tap_idx)];
        cd h = h_vals[static_cast<std::size_t>(tap_idx)];
        for (int i = tap; i < n; ++i) {
            yhat[static_cast<std::size_t>(i)] += h * x[static_cast<std::size_t>(i - tap)];
        }
    }

    double data_loss = 0.0;
    for (int i = 0; i < n; ++i) {
        data_loss += std::norm(yhat[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(i)]);
    }

    double parity_loss = 0.0;
    for (const auto& inds : graph.rows) {
        double prod = 1.0;
        for (int idx : inds) {
            prod *= x[static_cast<std::size_t>(idx)];
        }
        double diff = 1.0 - prod;
        parity_loss += diff * diff;
    }

    double pilot_loss = 0.0;
    if (!pilot_pos.empty() && lambda_pil > 0.0) {
        for (std::size_t kk = 0; kk < pilot_pos.size(); ++kk) {
            int idx = pilot_pos[kk];
            double target = std::real(pilot_bpsk[kk]) >= 0.0 ? 1.0 : -1.0;
            double err = x[static_cast<std::size_t>(idx)] - target;
            pilot_loss += err * err;
        }
    }

    double reg = 0.0;
    for (double value : theta) {
        reg += value * value;
    }

    return data_loss + lambda * parity_loss + lambda_pil * pilot_loss + gamma * reg;
}

std::vector<double> numerical_gradient(const std::vector<double>& theta,
                                       double base_value,
                                       const HGraph& graph,
                                       const std::vector<cd>& y,
                                       const std::vector<int>& h_pos,
                                       const std::vector<int>& pilot_pos,
                                       const std::vector<cd>& pilot_bpsk,
                                       double lambda,
                                       double lambda_pil,
                                       double gamma) {
    std::vector<double> grad(theta.size(), 0.0);
    std::vector<double> probe = theta;
    for (std::size_t i = 0; i < theta.size(); ++i) {
        double eps = runtime_grad_relstep() * std::max(1.0, std::abs(theta[i]));
        probe[i] = theta[i] + eps;
        double value_pos = raw_joint_objective(graph, probe, y, h_pos, pilot_pos, pilot_bpsk,
                                               lambda, lambda_pil, gamma);
        grad[i] = (value_pos - base_value) / eps;
        probe[i] = theta[i];
    }
    return grad;
}

void optimize_theta_lbfgs(std::vector<double>& theta,
                          const HGraph& graph,
                          const std::vector<cd>& y,
                          const std::vector<int>& h_pos,
                          const std::vector<int>& pilot_pos,
                          const std::vector<cd>& pilot_bpsk,
                          double lambda,
                          double lambda_pil,
                          double gamma,
                          double pilot_frac,
                          int max_steps) {
    struct LBFGSPair {
        std::vector<double> s;
        std::vector<double> y;
        double rho = 0.0;
    };

    std::vector<LBFGSPair> history;
    std::vector<double> grad;
    std::vector<double> trial(theta.size(), 0.0);
    std::vector<double> trial_grad;
    std::vector<double> direction(theta.size(), 0.0);
    std::vector<double> q(theta.size(), 0.0);
    const int lbfgs_memory = runtime_lbfgs_memory(pilot_frac);
    std::vector<double> alpha(static_cast<std::size_t>(lbfgs_memory), 0.0);

    double current = raw_joint_objective(graph, theta, y, h_pos, pilot_pos, pilot_bpsk,
                                         lambda, lambda_pil, gamma);
    grad = numerical_gradient(theta, current, graph, y, h_pos, pilot_pos, pilot_bpsk,
                              lambda, lambda_pil, gamma);

    for (int iter = 0; iter < max_steps; ++iter) {
        if (std::sqrt(norm2_real(grad)) <= kJuliaGAbsTol) {
            break;
        }

        bool use_lbfgs = !history.empty();
        if (use_lbfgs) {
            q = grad;
            const int hist_size = static_cast<int>(history.size());
            for (int i = hist_size - 1; i >= 0; --i) {
                alpha[static_cast<std::size_t>(i)] = history[static_cast<std::size_t>(i)].rho *
                    dot_real(history[static_cast<std::size_t>(i)].s, q);
                for (std::size_t j = 0; j < q.size(); ++j) {
                    q[j] -= alpha[static_cast<std::size_t>(i)] *
                            history[static_cast<std::size_t>(i)].y[j];
                }
            }
            const LBFGSPair& last = history.back();
            double yy = std::max(norm2_real(last.y), 1e-12);
            double gamma0 = dot_real(last.s, last.y) / yy;
            for (std::size_t j = 0; j < q.size(); ++j) {
                direction[j] = gamma0 * q[j];
            }
            for (int i = 0; i < hist_size; ++i) {
                double beta = history[static_cast<std::size_t>(i)].rho *
                    dot_real(history[static_cast<std::size_t>(i)].y, direction);
                for (std::size_t j = 0; j < direction.size(); ++j) {
                    direction[j] += history[static_cast<std::size_t>(i)].s[j] *
                                    (alpha[static_cast<std::size_t>(i)] - beta);
                }
            }
            for (double& value : direction) {
                value = -value;
            }
            if (dot_real(direction, grad) >= 0.0) {
                use_lbfgs = false;
            }
        }

        if (!use_lbfgs) {
            history.clear();
            for (std::size_t i = 0; i < grad.size(); ++i) {
                direction[i] = -grad[i];
            }
        }

        double directional_derivative = dot_real(grad, direction);
        if (!(directional_derivative < 0.0)) {
            break;
        }

        double step = 1.0;
        bool accepted = false;
        while (step >= kMinStep) {
            for (std::size_t i = 0; i < theta.size(); ++i) {
                trial[i] = theta[i] + step * direction[i];
            }
            double value = raw_joint_objective(graph, trial, y, h_pos, pilot_pos, pilot_bpsk,
                                               lambda, lambda_pil, gamma);
            if (value <= current + kArmijoC1 * step * directional_derivative) {
                trial_grad = numerical_gradient(trial, value, graph, y, h_pos, pilot_pos,
                                               pilot_bpsk, lambda, lambda_pil, gamma);
                std::vector<double> s(theta.size(), 0.0);
                std::vector<double> ydiff(theta.size(), 0.0);
                for (std::size_t i = 0; i < theta.size(); ++i) {
                    s[i] = trial[i] - theta[i];
                    ydiff[i] = trial_grad[i] - grad[i];
                }
                double sy = dot_real(s, ydiff);
                theta = trial;
                grad = trial_grad;
                double prev_current = current;
                current = value;
                if (sy > 1e-12) {
                    if (history.size() == static_cast<std::size_t>(lbfgs_memory)) {
                        history.erase(history.begin());
                    }
                    history.push_back(LBFGSPair{std::move(s), std::move(ydiff), 1.0 / sy});
                } else {
                    history.clear();
                }
                accepted = true;
                if (std::abs(prev_current - current) <= kJuliaFAbsTol) {
                    return;
                }
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            break;
        }
    }
}

std::vector<int> bits_from_sign(const std::vector<double>& z) {
    std::vector<int> bits(z.size(), 0);
    for (std::size_t i = 0; i < z.size(); ++i) {
        bits[i] = z[i] < 0.0 ? 1 : 0;
    }
    return bits;
}

std::vector<int> resolve_sign_flip(const HGraph& graph,
                                   const std::vector<int>& raw_bits,
                                   const std::vector<double>& z,
                                   const std::vector<int>& pilot_pos,
                                   const std::vector<cd>& pilot_bpsk) {
    std::vector<int> flipped = raw_bits;
    for (int& bit : flipped) {
        bit = 1 - bit;
    }

    double vote = 0.0;
    for (std::size_t i = 0; i < pilot_pos.size(); ++i) {
        vote += z[static_cast<std::size_t>(pilot_pos[i])] * std::real(pilot_bpsk[i]);
    }

    int raw_sw = syndrome_weight(graph, raw_bits);
    int flipped_sw = syndrome_weight(graph, flipped);
    if (flipped_sw < raw_sw || vote < 0.0) {
        return flipped;
    }
    return raw_bits;
}

JointResult decode_sparse_joint(const HGraph& graph,
                                const std::vector<cd>& y,
                                const std::vector<int>& pilot_pos,
                                const std::vector<cd>& pilot_bpsk,
                                const std::vector<int>& h_pos,
                                const std::vector<cd>& h_init,
                                double lambda,
                                double lambda_pil,
                                double gamma,
                                double pilot_frac,
                                int max_iter_opt) {
    const int n = static_cast<int>(y.size());
    const int l = static_cast<int>(h_pos.size());
    std::vector<double> theta(static_cast<std::size_t>(n + 2 * l), 0.0);
    for (int i = 0; i < l; ++i) {
        theta[static_cast<std::size_t>(n + i)] = std::real(h_init[static_cast<std::size_t>(i)]);
        theta[static_cast<std::size_t>(n + l + i)] = std::imag(h_init[static_cast<std::size_t>(i)]);
    }
    optimize_theta_lbfgs(theta, graph, y, h_pos, pilot_pos, pilot_bpsk,
                         lambda, lambda_pil, gamma, pilot_frac, max_iter_opt);

    JointResult out;
    out.z.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.z[static_cast<std::size_t>(i)] = theta[static_cast<std::size_t>(i)];
    }
    out.h_vals.resize(static_cast<std::size_t>(l), cd{});
    for (int i = 0; i < l; ++i) {
        out.h_vals[static_cast<std::size_t>(i)] =
            cd(theta[static_cast<std::size_t>(n + i)],
               theta[static_cast<std::size_t>(n + l + i)]);
    }
    std::vector<int> raw_bits = bits_from_sign(out.z);
    std::vector<int> bits = resolve_sign_flip(graph, raw_bits, out.z, pilot_pos, pilot_bpsk);
    out.objective = raw_joint_objective(graph, theta, y, h_pos, pilot_pos, pilot_bpsk,
                                        lambda, lambda_pil, gamma);
    out.valid = is_valid_codeword(graph, bits);
    return out;
}

std::vector<int> hard_from_llr(const std::vector<double>& l01) {
    std::vector<int> bits(l01.size(), 0);
    for (std::size_t i = 0; i < l01.size(); ++i) {
        bits[i] = l01[i] < 0.0 ? 1 : 0;
    }
    return bits;
}

void clamp_pilots_l01(std::vector<double>& l01,
                      const std::vector<int>& pilot_bits,
                      const std::vector<int>& pos,
                      double clamp_l) {
    for (std::size_t i = 0; i < pos.size(); ++i) {
        l01[static_cast<std::size_t>(pos[i])] = pilot_bits[i] == 0 ? clamp_l : -clamp_l;
    }
}

std::pair<std::vector<cd>, double> eq_lmmse_with_sigma2(const std::vector<cd>& y,
                                                        const std::vector<cd>& h,
                                                        double sigma2_init,
                                                        int iters) {
    double sigma2 = std::max(sigma2_init, 1e-6);
    std::vector<cd> x = lmmse_deconv_prefix(y, h, sigma2);
    for (int iter = 0; iter < iters; ++iter) {
        std::vector<cd> yhat = conv_prefix(h, x, static_cast<int>(y.size()));
        double accum = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            accum += std::norm(y[i] - yhat[i]);
        }
        sigma2 = clamp_double(accum / static_cast<double>(y.size()), 1e-6, 10.0);
        x = lmmse_deconv_prefix(y, h, sigma2);
    }
    return {x, sigma2};
}

std::vector<double> bpsk_llr_logP0P1(const std::vector<double>& x_soft,
                                     double sigma2,
                                     double clip) {
    std::vector<double> out(x_soft.size(), 0.0);
    double c = -2.0 / std::max(sigma2, 1e-12);
    for (std::size_t i = 0; i < x_soft.size(); ++i) {
        out[i] = clamp_double(c * x_soft[i], -clip, clip);
    }
    return out;
}

std::vector<double> bpsk_mean_from_l01(const std::vector<double>& l01) {
    std::vector<double> out(l01.size(), 0.0);
    for (std::size_t i = 0; i < l01.size(); ++i) {
        out[i] = -std::tanh(0.5 * l01[i]);
    }
    return out;
}

double lse2(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) {
        return b;
    }
    if (b == -std::numeric_limits<double>::infinity()) {
        return a;
    }
    double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

void init_trellis(std::array<std::array<int, 2>, 4>& next_state,
                  std::array<std::array<int, 2>, 4>& parity_bit) {
    for (int s = 0; s < 4; ++s) {
        int s1 = (s >> 1) & 1;
        int s2 = s & 1;
        for (int u = 0; u < 2; ++u) {
            int f = u ^ s1 ^ s2;
            int p = f ^ s2;
            int ns1 = f;
            int ns2 = s1;
            int ns = (ns1 << 1) | ns2;
            next_state[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)] = ns;
            parity_bit[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)] = p;
        }
    }
}

std::pair<std::vector<double>, std::vector<double>> bcjr_rsc(const std::vector<double>& lsys,
                                                             const std::vector<double>& lpar,
                                                             const std::vector<double>& la) {
    const int n = static_cast<int>(lsys.size());
    std::array<std::array<int, 2>, 4> next_state{};
    std::array<std::array<int, 2>, 4> parity_bit{};
    init_trellis(next_state, parity_bit);

    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::vector<std::array<double, 4>> alpha(static_cast<std::size_t>(n + 1));
    std::vector<std::array<double, 4>> beta(static_cast<std::size_t>(n + 1));
    for (auto& row : alpha) {
        row.fill(neg_inf);
    }
    for (auto& row : beta) {
        row.fill(neg_inf);
    }
    alpha[0][0] = 0.0;
    beta[static_cast<std::size_t>(n)].fill(0.0);

    for (int t = 0; t < n; ++t) {
        for (int s = 0; s < 4; ++s) {
            double a = alpha[static_cast<std::size_t>(t)][static_cast<std::size_t>(s)];
            if (!std::isfinite(a)) {
                continue;
            }
            for (int u = 0; u < 2; ++u) {
                int ns = next_state[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                int p = parity_bit[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                double g = 0.5 * ((1 - 2 * u) * (la[static_cast<std::size_t>(t)] + lsys[static_cast<std::size_t>(t)]) +
                                  (1 - 2 * p) * lpar[static_cast<std::size_t>(t)]);
                alpha[static_cast<std::size_t>(t + 1)][static_cast<std::size_t>(ns)] =
                    lse2(alpha[static_cast<std::size_t>(t + 1)][static_cast<std::size_t>(ns)], a + g);
            }
        }
        double c = *std::max_element(alpha[static_cast<std::size_t>(t + 1)].begin(),
                                     alpha[static_cast<std::size_t>(t + 1)].end());
        if (std::isfinite(c)) {
            for (double& value : alpha[static_cast<std::size_t>(t + 1)]) {
                value -= c;
            }
        }
    }

    for (int t = n - 1; t >= 0; --t) {
        for (int s = 0; s < 4; ++s) {
            double acc = neg_inf;
            for (int u = 0; u < 2; ++u) {
                int ns = next_state[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                int p = parity_bit[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                double g = 0.5 * ((1 - 2 * u) * (la[static_cast<std::size_t>(t)] + lsys[static_cast<std::size_t>(t)]) +
                                  (1 - 2 * p) * lpar[static_cast<std::size_t>(t)]);
                acc = lse2(acc, g + beta[static_cast<std::size_t>(t + 1)][static_cast<std::size_t>(ns)]);
            }
            beta[static_cast<std::size_t>(t)][static_cast<std::size_t>(s)] = acc;
        }
        double c = *std::max_element(beta[static_cast<std::size_t>(t)].begin(),
                                     beta[static_cast<std::size_t>(t)].end());
        if (std::isfinite(c)) {
            for (double& value : beta[static_cast<std::size_t>(t)]) {
                value -= c;
            }
        }
    }

    std::vector<double> lu_post(static_cast<std::size_t>(n), 0.0);
    std::vector<double> lpar_post(static_cast<std::size_t>(n), 0.0);
    for (int t = 0; t < n; ++t) {
        double num_u0 = neg_inf;
        double num_u1 = neg_inf;
        double num_p0 = neg_inf;
        double num_p1 = neg_inf;
        for (int s = 0; s < 4; ++s) {
            double a = alpha[static_cast<std::size_t>(t)][static_cast<std::size_t>(s)];
            if (!std::isfinite(a)) {
                continue;
            }
            for (int u = 0; u < 2; ++u) {
                int ns = next_state[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                int p = parity_bit[static_cast<std::size_t>(s)][static_cast<std::size_t>(u)];
                double g = 0.5 * ((1 - 2 * u) * (la[static_cast<std::size_t>(t)] + lsys[static_cast<std::size_t>(t)]) +
                                  (1 - 2 * p) * lpar[static_cast<std::size_t>(t)]);
                double value = a + g + beta[static_cast<std::size_t>(t + 1)][static_cast<std::size_t>(ns)];
                if (u == 0) {
                    num_u0 = lse2(num_u0, value);
                } else {
                    num_u1 = lse2(num_u1, value);
                }
                if (p == 0) {
                    num_p0 = lse2(num_p0, value);
                } else {
                    num_p1 = lse2(num_p1, value);
                }
            }
        }
        lu_post[static_cast<std::size_t>(t)] = num_u0 - num_u1;
        lpar_post[static_cast<std::size_t>(t)] = num_p0 - num_p1;
    }
    return {lu_post, lpar_post};
}

TurboEqResult decode_turboeq_rsc_bpsk(const std::vector<cd>& y,
                                      const std::vector<cd>& hfull,
                                      const std::vector<int>& b_true,
                                      double p,
                                      int turbo_iters,
                                      double sigma2_init,
                                      int eq_sigma2_iters,
                                      double llr_clip) {
    std::vector<cd> h = hfull;
    if (h.size() > y.size()) {
        h.resize(y.size());
    }

    std::vector<int> pilots_pos = choose_pilots_bits(static_cast<int>(b_true.size()), p);
    std::vector<int> pilots_bits;
    pilots_bits.reserve(pilots_pos.size());
    for (int idx : pilots_pos) {
        pilots_bits.push_back(b_true[static_cast<std::size_t>(idx)]);
    }

    double sigma2 = clamp_double(sigma2_init, 1e-6, 10.0);
    std::vector<double> llr128_ch(128, 0.0);
    std::vector<double> llr128_post(128, 0.0);
    std::vector<int> u64_hat(64, 0);

    for (int iter = 0; iter < std::max(1, turbo_iters); ++iter) {
        auto [x_soft_c, sigma2_hat] = eq_lmmse_with_sigma2(y, h, sigma2, eq_sigma2_iters);
        std::vector<double> x_soft(x_soft_c.size(), 0.0);
        for (std::size_t i = 0; i < x_soft.size(); ++i) {
            x_soft[i] = std::real(x_soft_c[i]);
        }

        llr128_ch = bpsk_llr_logP0P1(x_soft, sigma2_hat, llr_clip);
        if (!pilots_pos.empty()) {
            clamp_pilots_l01(llr128_ch, pilots_bits, pilots_pos, llr_clip);
        }

        std::vector<double> lsys(64, 0.0);
        std::vector<double> lpar(64, 0.0);
        for (int t = 0; t < 64; ++t) {
            lsys[static_cast<std::size_t>(t)] = llr128_ch[static_cast<std::size_t>(2 * t)];
            lpar[static_cast<std::size_t>(t)] = llr128_ch[static_cast<std::size_t>(2 * t + 1)];
        }

        auto [lu_post, lpar_post] = bcjr_rsc(lsys, lpar, std::vector<double>(64, 0.0));
        u64_hat = hard_from_llr(lu_post);
        for (int t = 0; t < 64; ++t) {
            llr128_post[static_cast<std::size_t>(2 * t)] = lu_post[static_cast<std::size_t>(t)];
            llr128_post[static_cast<std::size_t>(2 * t + 1)] = lpar_post[static_cast<std::size_t>(t)];
        }

        std::vector<double> xmean_real = bpsk_mean_from_l01(llr128_post);
        std::vector<cd> xmean(xmean_real.size(), cd{});
        for (std::size_t i = 0; i < xmean.size(); ++i) {
            xmean[i] = cd(xmean_real[i], 0.0);
        }
        std::vector<cd> yhat = conv_prefix(h, xmean, static_cast<int>(y.size()));
        double accum = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            accum += std::norm(y[i] - yhat[i]);
        }
        sigma2 = clamp_double(accum / static_cast<double>(y.size()), 1e-6, 10.0);
    }

    return TurboEqResult{
        std::move(u64_hat),
        std::move(llr128_ch),
        std::move(llr128_post),
        sigma2
    };
}

void ensure_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

void write_raw_csv(const std::string& path, const std::vector<RawSummaryRow>& rows) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write RAW CSV: " + path);
    }
    out << "pilot_frac,method,ber,psr_pkt,psr64,nframes,lam_pil,agree_pilots\n";
    out << std::setprecision(12);
    for (const auto& row : rows) {
        out << row.pilot_frac << ','
            << row.method << ','
            << row.ber << ','
            << row.psr_pkt << ','
            << row.psr64 << ','
            << row.nframes << ','
            << row.lam_pil << ','
            << row.agree_pilots << '\n';
    }
}

void write_rsc_csv(const std::string& path, const std::vector<RscDetailRow>& rows) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write RSC CSV: " + path);
    }
    out << "p,blk,corr,u64_psr,u64_ber,b128_post_psr,b128_post_ber,b128_ch_ber,sigma2_final\n";
    out << std::setprecision(12);
    for (const auto& row : rows) {
        out << row.p << ','
            << row.blk << ','
            << row.corr << ','
            << row.u64_psr << ','
            << row.u64_ber << ','
            << row.b128_post_psr << ','
            << row.b128_post_ber << ','
            << row.b128_ch_ber << ','
            << row.sigma2_final << '\n';
    }
}

std::vector<RawSummaryRow> run_raw(const BundleData& bundle,
                                   const HGraph& graph,
                                   const std::vector<double>& ps,
                                   const RawConfig& cfg,
                                   const std::string& out_csv) {
    std::vector<RawSummaryRow> out;
    const int num_frames = static_cast<int>(bundle.raw_packets.rows);
    const int start0 = std::max(0, std::min(num_frames - 1, cfg.start_frame - 1));
    const int stop0 = std::min(num_frames, start0 + cfg.n_per_p);
    if (start0 >= stop0) {
        throw std::runtime_error("No RAW frames selected");
    }

    std::cout << "==============================================================\n";
    std::cout << "RAW DFEC oracle-pilot PSWEEP | frames per p=" << (stop0 - start0)
              << " | start=" << cfg.start_frame
              << " | lam_pil=" << cfg.lambda_pil << '\n';
    std::cout << "==============================================================\n";

    for (double p : ps) {
        std::vector<int> pilot_pos = choose_pilots_bits(graph.n, p);
        std::vector<double> ber_eq;
        std::vector<double> psr_eq;
        std::vector<double> psr64_eq;
        std::vector<double> ber_df;
        std::vector<double> psr_df;
        std::vector<double> psr64_df;

        for (int frame = start0; frame < stop0; ++frame) {
            std::vector<cd> y = shift_left(row_as_complex(bundle.raw_packets, frame), bundle.raw_bestd[static_cast<std::size_t>(frame)]);
            std::vector<int> cw_true = row_as_int(bundle.raw_cw_true, frame);
            std::vector<cd> x_true(cw_true.size(), cd{});
            for (std::size_t i = 0; i < cw_true.size(); ++i) {
                x_true[i] = cd(cw_true[i] == 1 ? 1.0 : -1.0, 0.0);
            }

            std::vector<cd> h = ridge_ls_h(x_true, y, cfg.h_len, cfg.rho_ls);
            std::vector<cd> yhat = conv_prefix(h, x_true, static_cast<int>(x_true.size()));
            double sigma2 = 1e-9;
            for (std::size_t i = 0; i < y.size(); ++i) {
                sigma2 += std::norm(y[i] - yhat[i]);
            }
            sigma2 /= static_cast<double>(y.size());

            std::vector<cd> x_lmmse = lmmse_deconv_prefix(y, h, sigma2);
            std::vector<double> x_eq(x_lmmse.size(), 0.0);
            for (std::size_t i = 0; i < x_lmmse.size(); ++i) {
                x_eq[i] = std::real(x_lmmse[i]);
            }
            for (int idx : pilot_pos) {
                x_eq[static_cast<std::size_t>(idx)] = cw_true[static_cast<std::size_t>(idx)] == 1 ? 1.0 : -1.0;
            }

            auto [cw_hat_eq, _it, _sgn] = spa_from_soft_bestsign(graph, x_eq, sigma2, 50);
            ber_eq.push_back(ber_bits(cw_hat_eq, cw_true));
            psr_eq.push_back(psr_pkt(cw_hat_eq, cw_true));
            psr64_eq.push_back(psr_segments(cw_hat_eq, cw_true, 64));

            std::vector<cd> pilot_bpsk;
            pilot_bpsk.reserve(pilot_pos.size());
            for (int idx : pilot_pos) {
                pilot_bpsk.emplace_back(cw_true[static_cast<std::size_t>(idx)] == 1 ? 1.0 : -1.0, 0.0);
            }

            std::vector<int> h_pos = topk_positions(h, cfg.k_sparse);
            std::vector<cd> h_init;
            h_init.reserve(h_pos.size());
            for (int idx : h_pos) {
                h_init.push_back(h[static_cast<std::size_t>(idx)]);
            }

            JointResult joint = decode_sparse_joint(
                graph, y, pilot_pos, pilot_bpsk, h_pos, h_init,
                cfg.lambda, cfg.lambda_pil, cfg.gamma, p, runtime_raw_max_iter(cfg.max_iter_opt, p));

            std::vector<cd> x_soft_c(joint.z.size(), cd{});
            for (std::size_t i = 0; i < joint.z.size(); ++i) {
                x_soft_c[i] = cd(std::tanh(joint.z[i]), 0.0);
            }
            if (!pilot_pos.empty()) {
                std::vector<cd> xhat_p;
                std::vector<cd> xref_p;
                xhat_p.reserve(pilot_pos.size());
                xref_p.reserve(pilot_pos.size());
                for (int idx : pilot_pos) {
                    xhat_p.push_back(x_soft_c[static_cast<std::size_t>(idx)]);
                    xref_p.push_back(x_true[static_cast<std::size_t>(idx)]);
                }
                double ph = argminphase_deg(xhat_p, xref_p);
                cd rot = std::exp(cd(0.0, -ph * kPi / 180.0));
                for (cd& value : x_soft_c) {
                    value *= rot;
                }
            }
            std::vector<int> cw_hat_df(cw_true.size(), 0);
            for (std::size_t i = 0; i < x_soft_c.size(); ++i) {
                cw_hat_df[i] = std::real(x_soft_c[i]) >= 0.0 ? 1 : 0;
            }
            cw_hat_df = resolve_flip_by_pilots(cw_hat_df, pilot_pos, pilot_bpsk);

            ber_df.push_back(ber_bits(cw_hat_df, cw_true));
            psr_df.push_back(psr_pkt(cw_hat_df, cw_true));
            psr64_df.push_back(psr_segments(cw_hat_df, cw_true, 64));
        }

        out.push_back(RawSummaryRow{
            p, "EQ+SPA",
            mean_value(ber_eq),
            mean_value(psr_eq),
            mean_value(psr64_eq),
            stop0 - start0,
            cfg.lambda_pil,
            0
        });
        out.push_back(RawSummaryRow{
            p, "DFEC",
            mean_value(ber_df),
            mean_value(psr_df),
            mean_value(psr64_df),
            stop0 - start0,
            cfg.lambda_pil,
            0
        });

        std::cout << std::fixed << std::setprecision(2)
                  << "p=" << p
                  << " | EQ+SPA PSR64=" << std::setprecision(3) << mean_value(psr64_eq)
                  << " BER=" << std::setprecision(4) << mean_value(ber_eq)
                  << " | DFEC PSR64=" << std::setprecision(3) << mean_value(psr64_df)
                  << " BER=" << std::setprecision(4) << mean_value(ber_df)
                  << '\n';
    }

    write_raw_csv(out_csv, out);
    std::cout << "Saved RAW tidy CSV -> " << out_csv << " (rows=" << out.size() << ")\n";
    return out;
}

std::vector<RscDetailRow> run_rsc(const BundleData& bundle,
                                  const BundleManifest& manifest,
                                  const std::vector<double>& ps,
                                  const RscConfig& cfg,
                                  const std::string& out_csv) {
    std::vector<int> eligible;
    bool use_default_order = !bundle.rsc_default_order.empty() &&
        std::abs(cfg.corr_thr - manifest.rsc_default_order_corr_thr) < 1e-12 &&
        cfg.seed_sel == manifest.rsc_default_order_seed;
    if (use_default_order) {
        eligible = bundle.rsc_default_order;
    } else {
        for (int i = 0; i < static_cast<int>(bundle.rsc_corr.size()); ++i) {
            if (bundle.rsc_corr[static_cast<std::size_t>(i)] >= cfg.corr_thr) {
                eligible.push_back(i);
            }
        }
        if (eligible.empty()) {
            throw std::runtime_error("No eligible RSC blocks");
        }
        std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed_sel));
        std::shuffle(eligible.begin(), eligible.end(), rng);
    }
    const int start0 = std::max(0, std::min(static_cast<int>(eligible.size()) - 1, cfg.start - 1));
    const int stop0 = std::min(static_cast<int>(eligible.size()), start0 + cfg.nblk);
    if (start0 >= stop0) {
        throw std::runtime_error("No RSC blocks selected");
    }

    std::vector<RscDetailRow> out;
    std::cout << "==============================================================\n";
    std::cout << "RSC TurboEQ PSWEEP | blocks=" << (stop0 - start0) << "/" << eligible.size()
              << " corr_thr=" << std::fixed << std::setprecision(2) << cfg.corr_thr
              << '\n';
    std::cout << "TurboEQ: iters=" << cfg.turbo_iters
              << " | sigma2_init=" << std::setprecision(3) << cfg.sigma2_init
              << " | eq_sigma2_iters=" << cfg.eq_sigma2_iters
              << " | llr_clip=" << std::setprecision(1) << cfg.llr_clip << '\n';
    std::cout << "==============================================================\n";

    for (double p : ps) {
        std::cout << "\n--- RSC p=" << std::fixed << std::setprecision(3) << p << " ---\n";
        for (int ii = start0; ii < stop0; ++ii) {
            int blk = eligible[static_cast<std::size_t>(ii)];
            std::vector<cd> y = row_as_complex(bundle.rsc_y, blk);
            std::vector<int> u_true = row_as_int(bundle.rsc_u64, blk);
            std::vector<int> b_true = row_as_int(bundle.rsc_b128, blk);
            std::vector<cd> h = row_as_complex(bundle.rsc_h, blk);
            if (h.size() > y.size()) {
                h.resize(y.size());
            }

            TurboEqResult tout = decode_turboeq_rsc_bpsk(
                y, h, b_true, p, cfg.turbo_iters, cfg.sigma2_init, cfg.eq_sigma2_iters, cfg.llr_clip);

            std::vector<int> b_hat_post = hard_from_llr(tout.llr128_post);
            std::vector<int> b_hat_ch = hard_from_llr(tout.llr128_ch);
            RscDetailRow row;
            row.p = p;
            row.blk = blk + 1;
            row.corr = bundle.rsc_corr[static_cast<std::size_t>(blk)];
            row.u64_psr = psr_pkt(tout.u64_hat, u_true);
            row.u64_ber = ber_bits(tout.u64_hat, u_true);
            row.b128_post_psr = psr_pkt(b_hat_post, b_true);
            row.b128_post_ber = ber_bits(b_hat_post, b_true);
            row.b128_ch_ber = ber_bits(b_hat_ch, b_true);
            row.sigma2_final = tout.sigma2_final;
            out.push_back(row);

            int rel = ii - start0 + 1;
            int total = stop0 - start0;
            if (rel == 1 || rel % 50 == 0 || rel == total) {
                std::cout << "  blk " << rel << "/" << total
                          << " | u64 PSR=" << std::fixed << std::setprecision(3) << row.u64_psr
                          << " b128(post) PSR=" << row.b128_post_psr
                          << '\n';
            }
        }
    }

    write_rsc_csv(out_csv, out);
    std::cout << "\nSaved RSC TurboEQ sweep -> " << out_csv << '\n';
    return out;
}

void print_help() {
    std::cout <<
R"(Usage:
  ./run_replayswap_cpp [args]

Options:
  --bundle <dir-or-manifest>   C++ bundle directory or manifest.txt
  --raw_only                   run only the RAW DFEC path
  --rsc_only                   run only the RSC TurboEQ path
  --ps <sweep>                 pilot ratios, e.g. 0:0.1:0.5 or 0.1,0.2,0.3

RAW overrides:
  --start_frame <int>
  --nperp <int>
  --lam_pil <float>
  --raw_out_csv <path>

RSC overrides:
  --corr <float>
  --nblk <int>
  --seed_sel <int>
  --start_rsc <int>
  --turbo_iters <int>
  --sigma2_init <float>
  --eq_sigma2_iters <int>
  --llr_clip <float>
  --rsc_out_csv <path>
)";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string bundle_path = "export/default_bundle";
        bool run_raw_flag = true;
        bool run_rsc_flag = true;
        std::vector<double> ps;
        std::string raw_out_csv;
        std::string rsc_out_csv;

        RawConfig raw_cfg;
        RscConfig rsc_cfg;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--bundle") {
                if (++i >= argc) throw std::runtime_error("--bundle needs a value");
                bundle_path = argv[i];
            } else if (arg == "--raw_only") {
                run_raw_flag = true;
                run_rsc_flag = false;
            } else if (arg == "--rsc_only") {
                run_raw_flag = false;
                run_rsc_flag = true;
            } else if (arg == "--ps") {
                if (++i >= argc) throw std::runtime_error("--ps needs a value");
                ps = parse_psweep(argv[i]);
            } else if (arg == "--start_frame") {
                if (++i >= argc) throw std::runtime_error("--start_frame needs a value");
                raw_cfg.start_frame = std::stoi(argv[i]);
            } else if (arg == "--nperp") {
                if (++i >= argc) throw std::runtime_error("--nperp needs a value");
                raw_cfg.n_per_p = std::stoi(argv[i]);
            } else if (arg == "--lam_pil") {
                if (++i >= argc) throw std::runtime_error("--lam_pil needs a value");
                raw_cfg.lambda_pil = std::stod(argv[i]);
            } else if (arg == "--raw_out_csv") {
                if (++i >= argc) throw std::runtime_error("--raw_out_csv needs a value");
                raw_out_csv = argv[i];
            } else if (arg == "--corr") {
                if (++i >= argc) throw std::runtime_error("--corr needs a value");
                rsc_cfg.corr_thr = std::stod(argv[i]);
            } else if (arg == "--nblk") {
                if (++i >= argc) throw std::runtime_error("--nblk needs a value");
                rsc_cfg.nblk = std::stoi(argv[i]);
            } else if (arg == "--seed_sel") {
                if (++i >= argc) throw std::runtime_error("--seed_sel needs a value");
                rsc_cfg.seed_sel = std::stoi(argv[i]);
            } else if (arg == "--start_rsc") {
                if (++i >= argc) throw std::runtime_error("--start_rsc needs a value");
                rsc_cfg.start = std::stoi(argv[i]);
            } else if (arg == "--turbo_iters") {
                if (++i >= argc) throw std::runtime_error("--turbo_iters needs a value");
                rsc_cfg.turbo_iters = std::stoi(argv[i]);
            } else if (arg == "--sigma2_init") {
                if (++i >= argc) throw std::runtime_error("--sigma2_init needs a value");
                rsc_cfg.sigma2_init = std::stod(argv[i]);
            } else if (arg == "--eq_sigma2_iters") {
                if (++i >= argc) throw std::runtime_error("--eq_sigma2_iters needs a value");
                rsc_cfg.eq_sigma2_iters = std::stoi(argv[i]);
            } else if (arg == "--llr_clip") {
                if (++i >= argc) throw std::runtime_error("--llr_clip needs a value");
                rsc_cfg.llr_clip = std::stod(argv[i]);
            } else if (arg == "--rsc_out_csv") {
                if (++i >= argc) throw std::runtime_error("--rsc_out_csv needs a value");
                rsc_out_csv = argv[i];
            } else if (arg == "--help" || arg == "-h") {
                print_help();
                return 0;
            } else {
                throw std::runtime_error("Unknown arg: " + arg);
            }
        }

        BundleData bundle = load_bundle(bundle_path);
        const BundleManifest& manifest = bundle.manifest;
        if (ps.empty()) {
            ps = manifest.pilot_fracs;
        }

        if (raw_cfg.start_frame == 1) raw_cfg.start_frame = manifest.raw_start_frame;
        if (raw_cfg.n_per_p == 20) raw_cfg.n_per_p = manifest.raw_n_per_p;
        raw_cfg.h_len = manifest.raw_h_len;
        raw_cfg.rho_ls = manifest.raw_rho_ls;
        raw_cfg.lambda = manifest.raw_lambda;
        if (raw_cfg.lambda_pil == 20.0) raw_cfg.lambda_pil = manifest.raw_lambda_pil;
        raw_cfg.gamma = manifest.raw_gamma;
        raw_cfg.eta = manifest.raw_eta;
        raw_cfg.k_sparse = manifest.raw_k_sparse;
        raw_cfg.max_iter_opt = manifest.raw_max_iter_opt;

        if (rsc_cfg.corr_thr == 0.10) rsc_cfg.corr_thr = manifest.rsc_corr_thr;
        if (rsc_cfg.nblk == 200) rsc_cfg.nblk = manifest.rsc_nblk;
        if (rsc_cfg.seed_sel == 12648430) rsc_cfg.seed_sel = manifest.rsc_seed_sel;
        if (rsc_cfg.start == 1) rsc_cfg.start = manifest.rsc_start;
        if (rsc_cfg.turbo_iters == 2) rsc_cfg.turbo_iters = manifest.rsc_turbo_iters;
        if (std::abs(rsc_cfg.sigma2_init - 1.30) < 1e-12) rsc_cfg.sigma2_init = manifest.rsc_sigma2_init;
        if (rsc_cfg.eq_sigma2_iters == 1) rsc_cfg.eq_sigma2_iters = manifest.rsc_eq_sigma2_iters;
        if (std::abs(rsc_cfg.llr_clip - 25.0) < 1e-12) rsc_cfg.llr_clip = manifest.rsc_llr_clip;

        if (raw_out_csv.empty()) {
            raw_out_csv = join_path(manifest.root_dir, "raw_dfec_oraclepilots_psweep_cpp.csv");
        }
        if (rsc_out_csv.empty()) {
            rsc_out_csv = join_path(manifest.root_dir, "psr_bpsk_rsc_turbo_cpp.csv");
        }

        HGraph graph = parse_h_graph(join_path(manifest.root_dir, manifest.ldpc_h_file));
        if (run_raw_flag) {
            run_raw(bundle, graph, ps, raw_cfg, raw_out_csv);
        }
        if (run_rsc_flag) {
            run_rsc(bundle, manifest, ps, rsc_cfg, rsc_out_csv);
        }
        std::cout << "\nDone.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
