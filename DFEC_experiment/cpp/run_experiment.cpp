#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kArmijoC1 = 1e-4;
constexpr double kMinStep = 1e-6;
constexpr double kWarmClip = 0.98;
constexpr double kZLimit = 10.0;
constexpr double kGradTol = 1e-10;
constexpr int kOuterIters = 3;
constexpr int kZSteps = 20;
constexpr int kRestarts = 1;
constexpr int kLBFGSMemory = 6;

struct ComplexMatrix {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<cd> data;

    cd& operator()(std::int64_t r, std::int64_t c) {
        return data[static_cast<std::size_t>(r * cols + c)];
    }

    const cd& operator()(std::int64_t r, std::int64_t c) const {
        return data[static_cast<std::size_t>(r * cols + c)];
    }
};

struct IntVector {
    std::vector<int> values;
};

struct HGraph {
    int m = 0;
    int n = 0;
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> cols;
};

struct Manifest {
    std::string data_dir;
    std::string ldpc_h_file;
    int num_frames_to_process = 1;
    std::vector<double> pilot_fracs;
    int h_len = 40;
    int k_sparse = 4;
    double noise_variance = 1e-2;
    int code_k = 64;
    int code_n = 128;
    double lambda = 2.0;
    double gamma = 1e-3;
    double eta = 1.0;
    std::string results_file = "ldpc_ber_cpp.csv";
};

struct PacketMapping {
    std::vector<int> packet_to_train_idx;
    std::vector<int> packet_to_xdata_idx;
};

struct JointResult {
    std::vector<int> bits;
    std::vector<double> z;
    std::vector<cd> h_vals;
    double objective = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct BPResult {
    std::vector<int> bits;
    int iters = 0;
    bool valid = false;
};

std::pair<std::vector<cd>, double> argminphase(const std::vector<cd>& xhat,
                                               const std::vector<cd>& xref);

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

std::string trim(const std::string& s) {
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
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

double clamp_double(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

ComplexMatrix read_complex_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open complex matrix: " + path);
    }

    ComplexMatrix mat;
    in.read(reinterpret_cast<char*>(&mat.rows), sizeof(mat.rows));
    in.read(reinterpret_cast<char*>(&mat.cols), sizeof(mat.cols));
    if (!in || mat.rows <= 0 || mat.cols <= 0) {
        throw std::runtime_error("Invalid complex matrix header: " + path);
    }

    mat.data.resize(static_cast<std::size_t>(mat.rows * mat.cols));
    for (cd& value : mat.data) {
        double re = 0.0;
        double im = 0.0;
        in.read(reinterpret_cast<char*>(&re), sizeof(re));
        in.read(reinterpret_cast<char*>(&im), sizeof(im));
        if (!in) {
            throw std::runtime_error("Short complex matrix payload: " + path);
        }
        value = cd(re, im);
    }
    return mat;
}

std::vector<int> read_int_vector(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open int vector: " + path);
    }
    std::int64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || count < 0) {
        throw std::runtime_error("Invalid int vector header: " + path);
    }
    std::vector<int> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        std::int64_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("Short int vector payload: " + path);
        }
        out[static_cast<std::size_t>(i)] = static_cast<int>(value);
    }
    return out;
}

Manifest read_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open manifest: " + path);
    }

    Manifest manifest;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (key == "data_dir") {
            manifest.data_dir = value;
        } else if (key == "ldpc_h_file") {
            manifest.ldpc_h_file = value;
        } else if (key == "num_frames_to_process") {
            manifest.num_frames_to_process = std::stoi(value);
        } else if (key == "pilot_fracs") {
            manifest.pilot_fracs.clear();
            for (const std::string& part : split(value, ',')) {
                manifest.pilot_fracs.push_back(std::stod(trim(part)));
            }
        } else if (key == "h_len") {
            manifest.h_len = std::stoi(value);
        } else if (key == "k_sparse") {
            manifest.k_sparse = std::stoi(value);
        } else if (key == "noise_variance") {
            manifest.noise_variance = std::stod(value);
        } else if (key == "code_k") {
            manifest.code_k = std::stoi(value);
        } else if (key == "code_n") {
            manifest.code_n = std::stoi(value);
        } else if (key == "lambda") {
            manifest.lambda = std::stod(value);
        } else if (key == "gamma") {
            manifest.gamma = std::stod(value);
        } else if (key == "eta") {
            manifest.eta = std::stod(value);
        } else if (key == "results_file") {
            manifest.results_file = value;
        }
    }

    if (manifest.pilot_fracs.empty()) {
        throw std::runtime_error("Manifest is missing pilot_fracs");
    }
    return manifest;
}

HGraph parse_h_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open LDPC H file: " + path);
    }

    std::map<int, std::vector<int>> row_map;
    std::string line;
    int max_col = -1;
    while (std::getline(in, line)) {
        std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        std::string row_text = trim(line.substr(0, colon));
        if (row_text.empty() || !std::isdigit(static_cast<unsigned char>(row_text[0]))) {
            continue;
        }

        int row = std::stoi(row_text);
        std::stringstream ss(line.substr(colon + 1));
        int col = 0;
        while (ss >> col) {
            row_map[row].push_back(col);
            max_col = std::max(max_col, col);
        }
    }

    HGraph graph;
    graph.m = row_map.empty() ? 0 : row_map.rbegin()->first + 1;
    graph.n = max_col + 1;
    graph.rows.assign(graph.m, {});
    graph.cols.assign(graph.n, {});
    for (const auto& [row, cols] : row_map) {
        graph.rows[row] = cols;
        for (int col : cols) {
            graph.cols[col].push_back(row);
        }
    }
    return graph;
}

template <typename T>
std::vector<T> gaussian_solve(std::vector<std::vector<T>> a, std::vector<T> b) {
    const int n = static_cast<int>(a.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::abs(a[col][col]);
        for (int row = col + 1; row < n; ++row) {
            double cand = std::abs(a[row][col]);
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot = row;
            }
        }
        if (pivot_abs < 1e-12) {
            throw std::runtime_error("Singular linear system");
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }
        for (int row = col + 1; row < n; ++row) {
            T factor = a[row][col] / a[col][col];
            a[row][col] = T{};
            for (int j = col + 1; j < n; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    std::vector<T> x(n);
    for (int row = n - 1; row >= 0; --row) {
        T sum = b[row];
        for (int j = row + 1; j < n; ++j) {
            sum -= a[row][j] * x[j];
        }
        x[row] = sum / a[row][row];
    }
    return x;
}

std::vector<cd> estimate_omp_channel(const std::vector<cd>& y_train,
                                     const std::vector<cd>& x_train,
                                     int h_len,
                                     int k_sparse) {
    const int n = static_cast<int>(y_train.size());
    std::vector<std::vector<cd>> X(n, std::vector<cd>(h_len, cd{}));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < h_len; ++j) {
            if (i - j >= 0) {
                X[i][j] = x_train[static_cast<std::size_t>(i - j)];
            }
        }
    }

    std::vector<cd> residual = y_train;
    std::vector<int> support;
    std::vector<bool> used(static_cast<std::size_t>(h_len), false);

    auto solve_support = [&](const std::vector<int>& active) {
        const int s = static_cast<int>(active.size());
        std::vector<std::vector<cd>> gram(s, std::vector<cd>(s, cd{}));
        std::vector<cd> rhs(s, cd{});
        for (int a = 0; a < s; ++a) {
            for (int b = 0; b < s; ++b) {
                cd sum{};
                for (int i = 0; i < n; ++i) {
                    sum += std::conj(X[i][active[a]]) * X[i][active[b]];
                }
                gram[a][b] = sum;
            }
            gram[a][a] += cd(1e-8, 0.0);
            cd sum{};
            for (int i = 0; i < n; ++i) {
                sum += std::conj(X[i][active[a]]) * y_train[static_cast<std::size_t>(i)];
            }
            rhs[a] = sum;
        }
        return gaussian_solve(gram, rhs);
    };

    for (int iter = 0; iter < k_sparse; ++iter) {
        int best_j = -1;
        double best_corr = -1.0;
        for (int j = 0; j < h_len; ++j) {
            if (used[static_cast<std::size_t>(j)]) {
                continue;
            }
            cd corr{};
            for (int i = 0; i < n; ++i) {
                corr += std::conj(X[i][j]) * residual[static_cast<std::size_t>(i)];
            }
            double mag = std::abs(corr);
            if (mag > best_corr) {
                best_corr = mag;
                best_j = j;
            }
        }
        if (best_j < 0) {
            break;
        }
        support.push_back(best_j);
        used[static_cast<std::size_t>(best_j)] = true;
        std::vector<cd> coeffs = solve_support(support);
        residual = y_train;
        for (int i = 0; i < n; ++i) {
            cd yhat{};
            for (std::size_t k = 0; k < support.size(); ++k) {
                yhat += X[i][support[k]] * coeffs[k];
            }
            residual[static_cast<std::size_t>(i)] -= yhat;
        }
    }

    std::vector<cd> h(static_cast<std::size_t>(h_len), cd{});
    if (!support.empty()) {
        std::vector<cd> coeffs = solve_support(support);
        for (std::size_t i = 0; i < support.size(); ++i) {
            h[static_cast<std::size_t>(support[i])] = coeffs[i];
        }
    }
    return h;
}

PacketMapping build_packet_mappings(const std::vector<int>& y_train_frames,
                                    const std::vector<int>& packet_frames,
                                    const std::vector<int>& packet_blocks,
                                    std::int64_t x_datas_rows) {
    std::unordered_map<int, int> frame_to_train;
    for (std::size_t i = 0; i < y_train_frames.size(); ++i) {
        frame_to_train[y_train_frames[i]] = static_cast<int>(i);
    }

    PacketMapping mapping;
    mapping.packet_to_train_idx.resize(packet_frames.size());
    for (std::size_t i = 0; i < packet_frames.size(); ++i) {
        auto it = frame_to_train.find(packet_frames[i]);
        if (it == frame_to_train.end()) {
            throw std::runtime_error("Missing training row for packet frame");
        }
        mapping.packet_to_train_idx[i] = it->second;
    }

    int num_frames = static_cast<int>(y_train_frames.size());
    if (num_frames <= 0 || x_datas_rows % num_frames != 0) {
        throw std::runtime_error("x_datas rows are not divisible by training rows");
    }
    int blocks_per_frame = static_cast<int>(x_datas_rows / num_frames);
    mapping.packet_to_xdata_idx.resize(packet_frames.size());
    for (std::size_t i = 0; i < packet_frames.size(); ++i) {
        mapping.packet_to_xdata_idx[i] =
            mapping.packet_to_train_idx[i] * blocks_per_frame + (packet_blocks[i] - 1);
    }
    return mapping;
}

std::vector<int> compute_pilot_positions(const HGraph& graph, double pilot_frac) {
    int start_row = static_cast<int>(std::llround((1.0 - pilot_frac) * graph.m)) + 1;
    start_row = std::max(1, std::min(start_row, graph.m));
    std::set<int> pilots;
    for (int row = start_row - 1; row < graph.m; ++row) {
        for (int col : graph.rows[row]) {
            pilots.insert(col);
        }
    }
    return std::vector<int>(pilots.begin(), pilots.end());
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
        weight += (sum != 0);
    }
    return weight;
}

cd dot_conj(const std::vector<cd>& a,
            const std::vector<cd>& b,
            std::size_t begin,
            std::size_t end) {
    cd sum{};
    for (std::size_t i = begin; i < end; ++i) {
        sum += std::conj(a[i]) * b[i];
    }
    return sum;
}

cd nearest_symbol(const std::vector<cd>& constellation, const cd& value) {
    if (constellation.empty()) {
        return value;
    }
    cd best = constellation.front();
    double best_dist = std::norm(value - best);
    for (std::size_t i = 1; i < constellation.size(); ++i) {
        double dist = std::norm(value - constellation[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = constellation[i];
        }
    }
    return best;
}

std::vector<cd> adaptive_dfe_equalize(const std::vector<cd>& y_train,
                                      const std::vector<cd>& x_train,
                                      const std::vector<cd>& y_data,
                                      const std::vector<cd>& x_ref,
                                      int fbsize,
                                      int pilot_len,
                                      double lambda = 0.99,
                                      double sigma = 1.0) {
    const int ffsize = 1;
    const int psize = ffsize + fbsize;
    const int tr_len = static_cast<int>(x_train.size());
    const int n_code = static_cast<int>(y_data.size());
    const int nsteps = tr_len + n_code;
    const int desired_len = std::min(tr_len + std::max(0, pilot_len), nsteps);
    const std::vector<cd> constellation = {cd(1.0, 0.0), cd(-1.0, 0.0)};

    std::vector<cd> p(static_cast<std::size_t>(psize), cd{});
    std::vector<cd> state(static_cast<std::size_t>(psize), cd{});
    std::vector<cd> out(static_cast<std::size_t>(nsteps), cd{});
    std::vector<cd> k(static_cast<std::size_t>(psize), cd{});
    std::vector<cd> dyRinv(static_cast<std::size_t>(psize), cd{});
    std::vector<cd> dy(static_cast<std::size_t>(psize), cd{});
    std::vector<std::vector<cd>> Rinv(static_cast<std::size_t>(psize), std::vector<cd>(static_cast<std::size_t>(psize), cd{}));
    for (int i = 0; i < psize; ++i) {
        Rinv[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = cd(sigma, 0.0);
    }

    for (int step = 0; step < nsteps; ++step) {
        cd xin = step < tr_len ? y_train[static_cast<std::size_t>(step)]
                               : y_data[static_cast<std::size_t>(step - tr_len)];

        state[0] = xin;
        cd yhat = dot_conj(p, state, 0, static_cast<std::size_t>(psize));
        out[static_cast<std::size_t>(step)] = yhat;
        dy = state;

        cd desired{};
        if (step < tr_len) {
            desired = x_train[static_cast<std::size_t>(step)];
        } else if (step < desired_len) {
            desired = x_ref[static_cast<std::size_t>(step - tr_len)];
        } else {
            desired = nearest_symbol(constellation, yhat);
        }

        cd error = desired - yhat;

        for (int row = 0; row < psize; ++row) {
            cd sum{};
            for (int col = 0; col < psize; ++col) {
                sum += Rinv[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] *
                       dy[static_cast<std::size_t>(col)];
            }
            k[static_cast<std::size_t>(row)] = sum;
        }
        cd denom(lambda, 0.0);
        for (int i = 0; i < psize; ++i) {
            denom += std::conj(dy[static_cast<std::size_t>(i)]) * k[static_cast<std::size_t>(i)];
        }
        if (std::abs(denom) < 1e-12) {
            denom = cd(1e-12, 0.0);
        }
        for (cd& value : k) {
            value /= denom;
        }

        for (int i = 0; i < psize; ++i) {
            p[static_cast<std::size_t>(i)] += k[static_cast<std::size_t>(i)] * std::conj(error);
        }

        for (int col = 0; col < psize; ++col) {
            cd sum{};
            for (int row = 0; row < psize; ++row) {
                sum += std::conj(dy[static_cast<std::size_t>(row)]) *
                       Rinv[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            }
            dyRinv[static_cast<std::size_t>(col)] = sum;
        }
        for (int row = 0; row < psize; ++row) {
            for (int col = 0; col < psize; ++col) {
                Rinv[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] -=
                    k[static_cast<std::size_t>(row)] * dyRinv[static_cast<std::size_t>(col)];
                Rinv[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] /= lambda;
            }
        }

        for (int idx = psize - 1; idx >= ffsize + 1; --idx) {
            state[static_cast<std::size_t>(idx)] = state[static_cast<std::size_t>(idx - 1)];
        }
        state[static_cast<std::size_t>(ffsize)] = desired;
    }

    std::vector<cd> equalized(static_cast<std::size_t>(n_code), cd{});
    for (int i = 0; i < n_code; ++i) {
        equalized[static_cast<std::size_t>(i)] = out[static_cast<std::size_t>(tr_len + i)];
    }
    auto [aligned, phase_deg] = argminphase(equalized, x_ref);
    (void) phase_deg;
    return aligned;
}

std::pair<std::vector<cd>, double> argminphase(const std::vector<cd>& xhat,
                                               const std::vector<cd>& xref) {
    double best_deg = 0.0;
    int best_err = std::numeric_limits<int>::max();
    std::vector<cd> best = xhat;
    for (int step = 0; step <= 3600; ++step) {
        double deg = 0.1 * step;
        cd rot = std::exp(cd(0.0, deg * kPi / 180.0));
        int err = 0;
        for (std::size_t i = 0; i < xhat.size(); ++i) {
            double xr = std::real(xhat[i] * rot);
            double rr = std::real(xref[i]);
            if ((xr >= 0.0) != (rr >= 0.0)) {
                ++err;
            }
        }
        if (err < best_err) {
            best_err = err;
            best_deg = deg;
            for (std::size_t i = 0; i < xhat.size(); ++i) {
                best[i] = xhat[i] * rot;
            }
        }
    }
    return {best, best_deg};
}

std::vector<double> warm_start_logits(const std::vector<cd>& xhat,
                                      const std::vector<int>& pilot,
                                      const std::vector<cd>& x_pilot,
                                      double clip = kWarmClip,
                                      double pilot_boost = 4.0) {
    std::vector<double> z0(xhat.size(), 0.0);
    for (std::size_t i = 0; i < xhat.size(); ++i) {
        z0[i] = std::atanh(clamp_double(std::real(xhat[i]), -clip, clip));
    }
    if (!pilot.empty()) {
        double pilot_score = 0.0;
        for (std::size_t i = 0; i < pilot.size(); ++i) {
            double a = std::real(xhat[static_cast<std::size_t>(pilot[i])]) >= 0.0 ? 1.0 : -1.0;
            double b = std::real(x_pilot[i]) >= 0.0 ? 1.0 : -1.0;
            pilot_score += a * b;
        }
        if (pilot_score < 0.0) {
            for (double& v : z0) {
                v = -v;
            }
        }
        for (std::size_t i = 0; i < pilot.size(); ++i) {
            z0[static_cast<std::size_t>(pilot[i])] = pilot_boost * (std::real(x_pilot[i]) >= 0.0 ? 1.0 : -1.0);
        }
    }
    return z0;
}

BPResult spa_decode(const HGraph& graph,
                    const std::vector<double>& y,
                    double sigma2,
                    int max_iter = 50) {
    const int n = graph.n;
    const int m = graph.m;
    std::vector<double> lch(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        lch[static_cast<std::size_t>(i)] = 2.0 * y[static_cast<std::size_t>(i)] / std::max(sigma2, 1e-12);
    }

    std::vector<std::vector<double>> q(static_cast<std::size_t>(m));
    std::vector<std::vector<double>> r(static_cast<std::size_t>(m));
    std::vector<std::vector<std::pair<int, int>>> col_entries(static_cast<std::size_t>(n));
    for (int row = 0; row < m; ++row) {
        q[static_cast<std::size_t>(row)].resize(graph.rows[row].size());
        r[static_cast<std::size_t>(row)].resize(graph.rows[row].size());
        for (std::size_t idx = 0; idx < graph.rows[row].size(); ++idx) {
            int col = graph.rows[row][idx];
            q[static_cast<std::size_t>(row)][idx] = lch[static_cast<std::size_t>(col)];
            col_entries[static_cast<std::size_t>(col)].push_back({row, static_cast<int>(idx)});
        }
    }

    BPResult out;
    out.bits.assign(static_cast<std::size_t>(n), 0);
    for (int iter = 1; iter <= max_iter; ++iter) {
        for (int row = 0; row < m; ++row) {
            const auto& cols = graph.rows[row];
            int d = static_cast<int>(cols.size());
            if (d == 0) {
                continue;
            }
            std::vector<double> tanhs(static_cast<std::size_t>(d), 0.0);
            std::vector<double> prefix(static_cast<std::size_t>(d), 1.0);
            std::vector<double> suffix(static_cast<std::size_t>(d), 1.0);
            for (int j = 0; j < d; ++j) {
                tanhs[static_cast<std::size_t>(j)] = std::tanh(0.5 * q[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)]);
            }
            for (int j = 1; j < d; ++j) {
                prefix[static_cast<std::size_t>(j)] = prefix[static_cast<std::size_t>(j - 1)] * tanhs[static_cast<std::size_t>(j - 1)];
            }
            for (int j = d - 2; j >= 0; --j) {
                suffix[static_cast<std::size_t>(j)] = suffix[static_cast<std::size_t>(j + 1)] * tanhs[static_cast<std::size_t>(j + 1)];
            }
            for (int j = 0; j < d; ++j) {
                double prod_except = clamp_double(prefix[static_cast<std::size_t>(j)] * suffix[static_cast<std::size_t>(j)], -0.999999, 0.999999);
                r[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] = 2.0 * std::atanh(prod_except);
            }
        }

        std::vector<double> lpost(static_cast<std::size_t>(n), 0.0);
        for (int col = 0; col < n; ++col) {
            double total = lch[static_cast<std::size_t>(col)];
            for (const auto& [row, idx] : col_entries[static_cast<std::size_t>(col)]) {
                total += r[static_cast<std::size_t>(row)][static_cast<std::size_t>(idx)];
            }
            lpost[static_cast<std::size_t>(col)] = total;
            out.bits[static_cast<std::size_t>(col)] = total < 0.0 ? 1 : 0;
            for (const auto& [row, idx] : col_entries[static_cast<std::size_t>(col)]) {
                q[static_cast<std::size_t>(row)][static_cast<std::size_t>(idx)] =
                    total - r[static_cast<std::size_t>(row)][static_cast<std::size_t>(idx)];
            }
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
                lambda * 2.0 * diff * (-prod_others) * (1.0 - x[static_cast<std::size_t>(global)] * x[static_cast<std::size_t>(global)]);
        }
    }
    return loss;
}

std::vector<cd> conv_supported(const std::vector<double>& x, const std::vector<cd>& h_full) {
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
                2.0 * std::real(res[static_cast<std::size_t>(i + tap)] * std::conj(h_vals[tap_idx]));
        }
    }
    return grad;
}

std::vector<cd> solve_supported_channel(const std::vector<double>& x,
                                        const std::vector<cd>& y,
                                        const std::vector<int>& h_pos,
                                        double gamma,
                                        double eta,
                                        const std::vector<cd>& h_prior) {
    const int n = static_cast<int>(x.size());
    const int l = static_cast<int>(h_pos.size());
    std::vector<std::vector<cd>> lhs(static_cast<std::size_t>(l), std::vector<cd>(static_cast<std::size_t>(l), cd{}));
    std::vector<cd> rhs(static_cast<std::size_t>(l), cd{});

    for (int row = 0; row < l; ++row) {
        rhs[static_cast<std::size_t>(row)] = eta * h_prior[static_cast<std::size_t>(row)];
        for (int col = 0; col < l; ++col) {
            double sum = row == col ? gamma + eta : 0.0;
            int start = std::max(h_pos[static_cast<std::size_t>(row)], h_pos[static_cast<std::size_t>(col)]);
            for (int i = start; i < n; ++i) {
                sum += x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(row)])] *
                       x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(col)])];
            }
            lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = cd(sum, 0.0);
        }
        lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(row)] += cd(1e-8, 0.0);
        for (int i = h_pos[static_cast<std::size_t>(row)]; i < n; ++i) {
            rhs[static_cast<std::size_t>(row)] += x[static_cast<std::size_t>(i - h_pos[static_cast<std::size_t>(row)])] *
                                                  y[static_cast<std::size_t>(i)];
        }
    }
    return gaussian_solve(lhs, rhs);
}

double joint_objective(const HGraph& graph,
                       const std::vector<double>& z,
                       const std::vector<cd>& y,
                       const std::vector<int>& h_pos,
                       const std::vector<cd>& h_vals,
                       const std::vector<cd>& h_prior,
                       double lambda,
                       double gamma,
                       double eta) {
    const int n = static_cast<int>(z.size());
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = std::tanh(z[static_cast<std::size_t>(i)]);
    }

    std::vector<cd> h_full(static_cast<std::size_t>(n), cd{});
    for (std::size_t i = 0; i < h_pos.size(); ++i) {
        h_full[static_cast<std::size_t>(h_pos[i])] = h_vals[i];
    }
    std::vector<cd> yhat = conv_supported(x, h_full);
    double data_loss = 0.0;
    for (int i = 0; i < n; ++i) {
        data_loss += std::norm(yhat[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(i)]);
    }
    double parity_loss = parity_loss_and_grad(graph, x, lambda, nullptr);
    double reg = 0.0;
    for (double v : z) {
        reg += v * v;
    }
    for (const cd& h : h_vals) {
        reg += std::norm(h);
    }
    double prior_loss = 0.0;
    for (std::size_t i = 0; i < h_vals.size(); ++i) {
        prior_loss += std::norm(h_vals[i] - h_prior[i]);
    }
    return data_loss + lambda * parity_loss + gamma * reg + eta * prior_loss;
}

double evaluate_z_objective(const HGraph& graph,
                            const std::vector<double>& z,
                            const std::vector<cd>& y,
                            const std::vector<int>& h_pos,
                            const std::vector<cd>& h_vals,
                            double lambda,
                            double gamma,
                            std::vector<double>* grad) {
    const int n = static_cast<int>(z.size());
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = std::tanh(z[static_cast<std::size_t>(i)]);
    }

    std::vector<cd> h_full(static_cast<std::size_t>(n), cd{});
    for (std::size_t i = 0; i < h_pos.size(); ++i) {
        h_full[static_cast<std::size_t>(h_pos[i])] = h_vals[i];
    }
    std::vector<cd> yhat = conv_supported(x, h_full);
    std::vector<cd> res(static_cast<std::size_t>(n), cd{});
    double data_loss = 0.0;
    for (int i = 0; i < n; ++i) {
        res[static_cast<std::size_t>(i)] = yhat[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(i)];
        data_loss += std::norm(res[static_cast<std::size_t>(i)]);
    }

    double reg = 0.0;
    if (grad) {
        *grad = linear_conv_grad_x(res, h_pos, h_vals, n);
        for (int i = 0; i < n; ++i) {
            (*grad)[static_cast<std::size_t>(i)] =
                (*grad)[static_cast<std::size_t>(i)] * (1.0 - x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)]) +
                2.0 * gamma * z[static_cast<std::size_t>(i)];
            reg += z[static_cast<std::size_t>(i)] * z[static_cast<std::size_t>(i)];
        }
        double parity_loss = parity_loss_and_grad(graph, x, lambda, grad);
        return data_loss + lambda * parity_loss + gamma * reg;
    }

    for (double v : z) {
        reg += v * v;
    }
    double parity_loss = parity_loss_and_grad(graph, x, lambda, nullptr);
    return data_loss + lambda * parity_loss + gamma * reg;
}

void optimize_z(const HGraph& graph,
                std::vector<double>& z,
                const std::vector<cd>& y,
                const std::vector<int>& h_pos,
                const std::vector<cd>& h_vals,
                double lambda,
                double gamma) {
    struct LBFGSPair {
        std::vector<double> s;
        std::vector<double> y;
        double rho = 0.0;
    };

    std::vector<LBFGSPair> history;
    std::vector<double> grad;
    std::vector<double> trial(z.size(), 0.0);
    std::vector<double> trial_grad;
    std::vector<double> direction(z.size(), 0.0);
    std::vector<double> q(z.size(), 0.0);
    std::vector<double> alpha(static_cast<std::size_t>(kLBFGSMemory), 0.0);

    double current = evaluate_z_objective(graph, z, y, h_pos, h_vals, lambda, gamma, &grad);
    for (int iter = 0; iter < kZSteps; ++iter) {
        double grad_norm2 = norm2_real(grad);
        if (grad_norm2 <= kGradTol * static_cast<double>(z.size())) {
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
                    q[j] -= alpha[static_cast<std::size_t>(i)] * history[static_cast<std::size_t>(i)].y[j];
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
            for (double& v : direction) {
                v = -v;
            }
            if (dot_real(direction, grad) >= 0.0) {
                use_lbfgs = false;
            }
        }

        if (!use_lbfgs) {
            for (std::size_t i = 0; i < grad.size(); ++i) {
                direction[i] = -grad[i];
            }
            history.clear();
        }

        double directional_derivative = dot_real(grad, direction);
        if (!(directional_derivative < 0.0)) {
            break;
        }

        double step = 1.0;
        bool accepted = false;
        while (step >= kMinStep) {
            for (std::size_t i = 0; i < z.size(); ++i) {
                trial[i] = clamp_double(z[i] + step * direction[i], -kZLimit, kZLimit);
            }
            double value = evaluate_z_objective(graph, trial, y, h_pos, h_vals, lambda, gamma, &trial_grad);
            if (value <= current + kArmijoC1 * step * directional_derivative) {
                std::vector<double> s(z.size(), 0.0);
                std::vector<double> ydiff(z.size(), 0.0);
                for (std::size_t i = 0; i < z.size(); ++i) {
                    s[i] = trial[i] - z[i];
                    ydiff[i] = trial_grad[i] - grad[i];
                }
                double sy = dot_real(s, ydiff);
                z = trial;
                grad = trial_grad;
                current = value;
                if (sy > 1e-12) {
                    if (history.size() == static_cast<std::size_t>(kLBFGSMemory)) {
                        history.erase(history.begin());
                    }
                    history.push_back(LBFGSPair{std::move(s), std::move(ydiff), 1.0 / sy});
                } else {
                    history.clear();
                }
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            history.clear();
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
                                   const std::vector<int>& pilot,
                                   const std::vector<cd>& pilot_bpsk) {
    std::vector<int> flipped = raw_bits;
    for (int& bit : flipped) {
        bit = 1 - bit;
    }

    double vote = 0.0;
    for (std::size_t i = 0; i < pilot.size(); ++i) {
        vote += z[static_cast<std::size_t>(pilot[i])] * std::real(pilot_bpsk[i]);
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
                                const std::vector<int>& pilot,
                                const std::vector<cd>& pilot_bpsk,
                                const std::vector<int>& h_pos,
                                const std::vector<cd>& h_init,
                                const std::vector<double>& z_init,
                                double lambda,
                                double gamma,
                                double eta) {
    JointResult best;
    bool have_best = false;

    for (int restart = 0; restart < kRestarts; ++restart) {
        double scale = restart == 0 ? 1.0 : 0.5;
        std::vector<double> z = z_init;
        for (double& v : z) {
            v = clamp_double(v * scale, -kZLimit, kZLimit);
        }
        std::vector<cd> h_vals = h_init;

        for (int outer = 0; outer < kOuterIters; ++outer) {
            std::vector<double> x(z.size(), 0.0);
            for (std::size_t i = 0; i < z.size(); ++i) {
                x[i] = std::tanh(z[i]);
            }
            h_vals = solve_supported_channel(x, y, h_pos, gamma, eta, h_init);
            optimize_z(graph, z, y, h_pos, h_vals, lambda, gamma);
        }

        std::vector<int> raw_bits = bits_from_sign(z);
        std::vector<int> bits = resolve_sign_flip(graph, raw_bits, z, pilot, pilot_bpsk);
        double obj = joint_objective(graph, z, y, h_pos, h_vals, h_init, lambda, gamma, eta);
        bool valid = is_valid_codeword(graph, bits);
        if (!have_best ||
            (valid && !best.valid) ||
            (valid == best.valid && obj < best.objective)) {
            best.bits = bits;
            best.z = z;
            best.h_vals = h_vals;
            best.objective = obj;
            best.valid = valid;
            have_best = true;
        }
    }

    if (!have_best) {
        throw std::runtime_error("Joint decoder failed to produce a candidate");
    }
    return best;
}

std::vector<int> symbol_bits_from_bpsk(const std::vector<cd>& x) {
    std::vector<int> bits(x.size(), 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        bits[i] = std::real(x[i]) < 0.0 ? 1 : 0;
    }
    return bits;
}

std::vector<int> complement_indices(int n, const std::vector<int>& subset) {
    std::vector<char> is_subset(static_cast<std::size_t>(n), 0);
    for (int idx : subset) {
        if (0 <= idx && idx < n) {
            is_subset[static_cast<std::size_t>(idx)] = 1;
        }
    }
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!is_subset[static_cast<std::size_t>(i)]) {
            out.push_back(i);
        }
    }
    return out;
}

double ber_on_indices(const std::vector<int>& truth_bits,
                      const std::vector<int>& est_bits,
                      const std::vector<int>& indices,
                      int denom_n) {
    int errors = 0;
    for (int idx : indices) {
        if (truth_bits[static_cast<std::size_t>(idx)] != est_bits[static_cast<std::size_t>(idx)]) {
            ++errors;
        }
    }
    return static_cast<double>(errors) / static_cast<double>(std::max(1, denom_n));
}

double ber_symbols_on_indices(const std::vector<cd>& truth_symbols,
                              const std::vector<cd>& est_symbols,
                              const std::vector<int>& indices,
                              int denom_n) {
    int errors = 0;
    for (int idx : indices) {
        if ((std::real(truth_symbols[static_cast<std::size_t>(idx)]) >= 0.0) !=
            (std::real(est_symbols[static_cast<std::size_t>(idx)]) >= 0.0)) {
            ++errors;
        }
    }
    return static_cast<double>(errors) / static_cast<double>(std::max(1, denom_n));
}

std::vector<int> symbol_mismatch_indices(const std::vector<cd>& truth_symbols,
                                         const std::vector<cd>& est_symbols,
                                         const std::vector<int>& indices) {
    std::vector<int> out;
    for (int idx : indices) {
        if ((std::real(truth_symbols[static_cast<std::size_t>(idx)]) >= 0.0) !=
            (std::real(est_symbols[static_cast<std::size_t>(idx)]) >= 0.0)) {
            out.push_back(idx);
        }
    }
    return out;
}

void print_debug_packet_summary(int packet_idx,
                                double pilot_frac,
                                int train_idx,
                                int xdata_idx,
                                int frame_id,
                                int block_id,
                                const std::vector<int>& pilot,
                                const std::vector<int>& nonpilot,
                                const std::vector<cd>& h_full,
                                const std::vector<cd>& x_ref,
                                const std::vector<cd>& xhat,
                                double ber_dfe) {
    std::vector<int> mismatches = symbol_mismatch_indices(x_ref, xhat, nonpilot);
    std::cout << "DEBUG packet=" << (packet_idx + 1)
              << " frame=" << frame_id
              << " block=" << block_id
              << " pilot=" << std::fixed << std::setprecision(2) << pilot_frac
              << " train_idx0=" << train_idx
              << " xdata_idx0=" << xdata_idx
              << '\n';
    std::cout << "DEBUG pilot_count=" << pilot.size()
              << " nonpilot_count=" << nonpilot.size()
              << " dfe_errors=" << mismatches.size()
              << " ber_dfe=" << ber_dfe
              << '\n';
    std::cout << "DEBUG pilot_head=";
    for (std::size_t i = 0; i < std::min<std::size_t>(pilot.size(), 12); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << pilot[i];
    }
    std::cout << '\n';
    std::cout << "DEBUG mismatch_head=";
    for (std::size_t i = 0; i < std::min<std::size_t>(mismatches.size(), 12); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << mismatches[i];
    }
    std::cout << '\n';
    std::cout << "DEBUG h_support=";
    bool first = true;
    for (std::size_t i = 0; i < h_full.size(); ++i) {
        if (std::abs(h_full[i]) <= 1e-12) {
            continue;
        }
        if (!first) {
            std::cout << ' ';
        }
        first = false;
        std::cout << i << ":" << std::setprecision(6) << std::real(h_full[i]) << "+"
                  << std::imag(h_full[i]) << "i";
    }
    std::cout << '\n';
    std::cout << "DEBUG xhat_head=";
    for (std::size_t i = 0; i < std::min<std::size_t>(xhat.size(), 8); ++i) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << std::setprecision(6) << std::real(xhat[i]) << "+" << std::imag(xhat[i]) << "i";
    }
    std::cout << '\n';
}

std::vector<cd> row_as_vector(const ComplexMatrix& mat, int row, int max_cols = -1) {
    int cols = max_cols < 0 ? static_cast<int>(mat.cols) : std::min<int>(max_cols, static_cast<int>(mat.cols));
    std::vector<cd> out(static_cast<std::size_t>(cols));
    for (int c = 0; c < cols; ++c) {
        out[static_cast<std::size_t>(c)] = mat(row, c);
    }
    return out;
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (!a.empty() && a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string bundle_dir = argc >= 2 ? argv[1] : "DFEC_experiment/export/cpp_bundle";
        std::string manifest_path = join_path(bundle_dir, "manifest.txt");
        Manifest manifest = read_manifest(manifest_path);

        std::string output_csv = argc >= 3 ? argv[2] : join_path(bundle_dir, manifest.results_file);
        int debug_packet = -1;
        double debug_pilot = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> override_pilot_fracs;
        for (int argi = 3; argi < argc; ++argi) {
            std::string arg = argv[argi];
            if (arg == "--debug-packet" && argi + 1 < argc) {
                debug_packet = std::stoi(argv[++argi]);
            } else if (arg == "--debug-pilot" && argi + 1 < argc) {
                debug_pilot = std::stod(argv[++argi]);
            } else if (arg == "--pilot-fracs" && argi + 1 < argc) {
                override_pilot_fracs.clear();
                for (const std::string& part : split(argv[++argi], ',')) {
                    std::string item = trim(part);
                    if (!item.empty()) {
                        override_pilot_fracs.push_back(std::stod(item));
                    }
                }
            }
        }
        if (!override_pilot_fracs.empty()) {
            manifest.pilot_fracs = override_pilot_fracs;
        }
        ComplexMatrix y_train_matrix = read_complex_matrix(join_path(bundle_dir, "y_train_matrix.c64bin"));
        ComplexMatrix packet_matrix = read_complex_matrix(join_path(bundle_dir, "packet_matrix.c64bin"));
        ComplexMatrix x_train_mat = read_complex_matrix(join_path(bundle_dir, "x_train.c64bin"));
        ComplexMatrix x_datas = read_complex_matrix(join_path(bundle_dir, "x_datas.c64bin"));
        std::vector<int> y_train_frames = read_int_vector(join_path(bundle_dir, "y_train_frames.i64bin"));
        std::vector<int> packet_frames = read_int_vector(join_path(bundle_dir, "packet_frames.i64bin"));
        std::vector<int> packet_blocks = read_int_vector(join_path(bundle_dir, "packet_blocks.i64bin"));

        HGraph graph = parse_h_graph(manifest.ldpc_h_file);
        if (graph.n != manifest.code_n) {
            throw std::runtime_error("LDPC length mismatch between manifest and H file");
        }

        std::vector<cd> x_train = row_as_vector(x_train_mat, 0);
        PacketMapping mapping = build_packet_mappings(y_train_frames, packet_frames, packet_blocks, x_datas.rows);

        std::vector<std::vector<cd>> h_est_omp(static_cast<std::size_t>(y_train_matrix.rows));
        std::cout << "Estimating OMP channels for " << y_train_matrix.rows << " training rows...\n";
        for (int row = 0; row < y_train_matrix.rows; ++row) {
            std::vector<cd> y_train = row_as_vector(y_train_matrix, row, static_cast<int>(x_train.size()));
            h_est_omp[static_cast<std::size_t>(row)] =
                estimate_omp_channel(y_train, x_train, manifest.h_len, manifest.k_sparse);
        }

        std::ofstream csv(output_csv);
        if (!csv) {
            throw std::runtime_error("Cannot open output CSV: " + output_csv);
        }
        csv << "frame,block,ber_grad,ber_dfe,ber_spa,ber_min,pilot_frac\n";

        int packets_to_process = std::min(manifest.num_frames_to_process, static_cast<int>(packet_matrix.rows));
        std::cout << "Processing " << packets_to_process << " packet rows...\n";
        std::cout << "UI_RUN packets=" << packets_to_process << '\n';

        for (double pilot_frac : manifest.pilot_fracs) {
            std::vector<int> pilot = compute_pilot_positions(graph, pilot_frac);
            std::vector<int> nonpilot = complement_indices(graph.n, pilot);
            std::cout << "Pilot fraction " << std::fixed << std::setprecision(2) << pilot_frac
                      << " -> " << pilot.size() << " pilot positions\n";
            std::cout << "UI_PILOT pilot=" << std::fixed << std::setprecision(2) << pilot_frac
                      << " pilot_count=" << pilot.size()
                      << " total_packets=" << packets_to_process
                      << '\n';

            int success_packets = 0;
            double sum_dfec_ber = 0.0;
            double sum_dfe_ber = 0.0;

            for (int packet_idx = 0; packet_idx < packets_to_process; ++packet_idx) {
                int train_idx = mapping.packet_to_train_idx[static_cast<std::size_t>(packet_idx)];
                int xdata_idx = mapping.packet_to_xdata_idx[static_cast<std::size_t>(packet_idx)];
                int frame_id = packet_frames[static_cast<std::size_t>(packet_idx)];
                int block_id = packet_blocks[static_cast<std::size_t>(packet_idx)];

                std::vector<cd> y_train = row_as_vector(y_train_matrix, train_idx, static_cast<int>(x_train.size()));
                std::vector<cd> y_data = row_as_vector(packet_matrix, packet_idx, manifest.code_n);
                std::vector<cd> x_ref = row_as_vector(x_datas, xdata_idx, manifest.code_n);
                std::vector<cd> h_full = h_est_omp[static_cast<std::size_t>(train_idx)];

                std::vector<cd> xhat = adaptive_dfe_equalize(
                    y_train, x_train, y_data, x_ref, manifest.h_len, static_cast<int>(pilot.size())
                );
                double ber_dfe = ber_symbols_on_indices(x_ref, xhat, nonpilot, manifest.code_n);
                if (debug_packet == packet_idx + 1 &&
                    std::abs(pilot_frac - debug_pilot) <= 1e-9) {
                    print_debug_packet_summary(
                        packet_idx, pilot_frac, train_idx, xdata_idx, frame_id, block_id,
                        pilot, nonpilot, h_full, x_ref, xhat, ber_dfe
                    );
                }

                std::vector<double> y_bp(static_cast<std::size_t>(manifest.code_n), 0.0);
                for (int i = 0; i < manifest.code_n; ++i) {
                    y_bp[static_cast<std::size_t>(i)] = std::real(xhat[static_cast<std::size_t>(i)]);
                }
                BPResult spa = spa_decode(graph, y_bp, 1.0, 50);
                std::vector<int> truth_bits = symbol_bits_from_bpsk(x_ref);
                double ber_spa = ber_on_indices(truth_bits, spa.bits, nonpilot, manifest.code_n);

                std::vector<int> h_pos;
                std::vector<cd> h_init;
                for (int i = 0; i < manifest.h_len; ++i) {
                    if (std::abs(h_full[static_cast<std::size_t>(i)]) > 1e-12) {
                        h_pos.push_back(i);
                        h_init.push_back(h_full[static_cast<std::size_t>(i)]);
                    }
                }
                if (h_pos.empty()) {
                    h_pos.push_back(0);
                    h_init.push_back(cd(1.0, 0.0));
                }

                std::vector<cd> x_pilot;
                x_pilot.reserve(pilot.size());
                for (int idx : pilot) {
                    x_pilot.push_back(x_ref[static_cast<std::size_t>(idx)]);
                }
                std::vector<double> z_init = warm_start_logits(xhat, pilot, x_pilot);
                JointResult joint = decode_sparse_joint(
                    graph, y_data, pilot, x_pilot, h_pos, h_init, z_init,
                    manifest.lambda, manifest.gamma, manifest.eta
                );

                std::vector<cd> joint_soft(static_cast<std::size_t>(manifest.code_n), cd{});
                for (int i = 0; i < manifest.code_n; ++i) {
                    joint_soft[static_cast<std::size_t>(i)] = cd(std::tanh(joint.z[static_cast<std::size_t>(i)]), 0.0);
                }

                std::vector<cd> joint_pilot;
                joint_pilot.reserve(pilot.size());
                for (int idx : pilot) {
                    joint_pilot.push_back(joint_soft[static_cast<std::size_t>(idx)]);
                }
                auto [rotated_pilot, phase_deg] = argminphase(joint_pilot, x_pilot);
                (void) rotated_pilot;
                cd rot = std::exp(cd(0.0, phase_deg * kPi / 180.0));
                std::vector<cd> joint_rotated(static_cast<std::size_t>(manifest.code_n), cd{});
                for (int i = 0; i < manifest.code_n; ++i) {
                    joint_rotated[static_cast<std::size_t>(i)] = joint_soft[static_cast<std::size_t>(i)] * rot;
                }
                double ber_grad = ber_symbols_on_indices(x_ref, joint_rotated, nonpilot, manifest.code_n);
                double ber_min = std::min(ber_spa, ber_dfe);
                bool success = ber_grad <= 1e-12;
                if (success) {
                    success_packets += 1;
                }
                sum_dfec_ber += ber_grad;
                sum_dfe_ber += ber_dfe;

                csv << frame_id << ',' << block_id << ','
                    << std::setprecision(12) << ber_grad << ','
                    << ber_dfe << ','
                    << ber_spa << ','
                    << ber_min << ','
                    << pilot_frac << '\n';

                std::cout << "UI_PACKET pilot=" << std::fixed << std::setprecision(2) << pilot_frac
                          << " packet_index=" << (packet_idx + 1)
                          << " total_packets=" << packets_to_process
                          << " frame=" << frame_id
                          << " block=" << block_id
                          << " dfec_ber=" << std::setprecision(12) << ber_grad
                          << " dfe_ber=" << ber_dfe
                          << " spa_ber=" << ber_spa
                          << " min_ber=" << ber_min
                          << " success=" << (success ? 1 : 0)
                          << " joint_valid=" << (joint.valid ? 1 : 0)
                          << '\n';

                std::cout << "Frame " << frame_id
                          << " Block " << block_id
                          << " | DFEC BER = " << std::fixed << std::setprecision(4) << ber_grad
                          << " | DFE+FEC BER = " << ber_min
                          << " | joint valid = " << (joint.valid ? "yes" : "no")
                          << '\n';
            }

            double success_rate = packets_to_process > 0
                ? static_cast<double>(success_packets) / static_cast<double>(packets_to_process)
                : 0.0;
            double mean_dfec_ber = packets_to_process > 0 ? sum_dfec_ber / static_cast<double>(packets_to_process) : 0.0;
            double mean_dfe_ber = packets_to_process > 0 ? sum_dfe_ber / static_cast<double>(packets_to_process) : 0.0;
            std::cout << "UI_PILOT_DONE pilot=" << std::fixed << std::setprecision(2) << pilot_frac
                      << " processed_packets=" << packets_to_process
                      << " success_packets=" << success_packets
                      << " success_rate=" << std::setprecision(12) << success_rate
                      << " mean_dfec_ber=" << mean_dfec_ber
                      << " mean_dfe_ber=" << mean_dfe_ber
                      << '\n';
        }

        std::cout << "Saved C++ experiment results to " << output_csv << '\n';
        std::cout << "UI_DONE output_csv=" << output_csv << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
