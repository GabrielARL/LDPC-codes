#include "replayswap_bundle.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace replayswap_cpp {

namespace {

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

IntMatrix read_int_matrix(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open int matrix: " + path);
    }

    IntMatrix mat;
    in.read(reinterpret_cast<char*>(&mat.rows), sizeof(mat.rows));
    in.read(reinterpret_cast<char*>(&mat.cols), sizeof(mat.cols));
    if (!in || mat.rows <= 0 || mat.cols <= 0) {
        throw std::runtime_error("Invalid int matrix header: " + path);
    }

    mat.data.resize(static_cast<std::size_t>(mat.rows * mat.cols));
    for (int& value : mat.data) {
        std::int64_t raw = 0;
        in.read(reinterpret_cast<char*>(&raw), sizeof(raw));
        if (!in) {
            throw std::runtime_error("Short int matrix payload: " + path);
        }
        value = static_cast<int>(raw);
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

    std::vector<int> out(static_cast<std::size_t>(count), 0);
    for (int& value : out) {
        std::int64_t raw = 0;
        in.read(reinterpret_cast<char*>(&raw), sizeof(raw));
        if (!in) {
            throw std::runtime_error("Short int vector payload: " + path);
        }
        value = static_cast<int>(raw);
    }
    return out;
}

std::vector<double> read_double_vector(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open double vector: " + path);
    }

    std::int64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || count < 0) {
        throw std::runtime_error("Invalid double vector header: " + path);
    }

    std::vector<double> out(static_cast<std::size_t>(count), 0.0);
    for (double& value : out) {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("Short double vector payload: " + path);
        }
    }
    return out;
}

}  // namespace

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

BundleManifest read_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open manifest: " + path);
    }

    BundleManifest manifest;
    std::size_t slash = path.find_last_of('/');
    manifest.root_dir = slash == std::string::npos ? "." : path.substr(0, slash);

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

        if (key == "bundle_name") {
            manifest.bundle_name = value;
        } else if (key == "ldpc_h_file") {
            manifest.ldpc_h_file = value;
        } else if (key == "raw_packets_file") {
            manifest.raw_packets_file = value;
        } else if (key == "raw_frame_ids_file") {
            manifest.raw_frame_ids_file = value;
        } else if (key == "raw_block_ids_file") {
            manifest.raw_block_ids_file = value;
        } else if (key == "raw_bestD_file") {
            manifest.raw_bestd_file = value;
        } else if (key == "raw_cw_true_file") {
            manifest.raw_cw_true_file = value;
        } else if (key == "rsc_y_file") {
            manifest.rsc_y_file = value;
        } else if (key == "rsc_u64_file") {
            manifest.rsc_u64_file = value;
        } else if (key == "rsc_b128_file") {
            manifest.rsc_b128_file = value;
        } else if (key == "rsc_h_file") {
            manifest.rsc_h_file = value;
        } else if (key == "rsc_corr_file") {
            manifest.rsc_corr_file = value;
        } else if (key == "rsc_frame_ids_file") {
            manifest.rsc_frame_ids_file = value;
        } else if (key == "rsc_default_order_file") {
            manifest.rsc_default_order_file = value;
        } else if (key == "pilot_fracs") {
            manifest.pilot_fracs.clear();
            for (const std::string& part : split(value, ',')) {
                if (!trim(part).empty()) {
                    manifest.pilot_fracs.push_back(std::stod(trim(part)));
                }
            }
        } else if (key == "raw_start_frame") {
            manifest.raw_start_frame = std::stoi(value);
        } else if (key == "raw_n_per_p") {
            manifest.raw_n_per_p = std::stoi(value);
        } else if (key == "raw_h_len") {
            manifest.raw_h_len = std::stoi(value);
        } else if (key == "raw_rho_ls") {
            manifest.raw_rho_ls = std::stod(value);
        } else if (key == "raw_lambda") {
            manifest.raw_lambda = std::stod(value);
        } else if (key == "raw_lambda_pil") {
            manifest.raw_lambda_pil = std::stod(value);
        } else if (key == "raw_gamma") {
            manifest.raw_gamma = std::stod(value);
        } else if (key == "raw_eta") {
            manifest.raw_eta = std::stod(value);
        } else if (key == "raw_k_sparse") {
            manifest.raw_k_sparse = std::stoi(value);
        } else if (key == "raw_max_iter_opt") {
            manifest.raw_max_iter_opt = std::stoi(value);
        } else if (key == "rsc_corr_thr") {
            manifest.rsc_corr_thr = std::stod(value);
        } else if (key == "rsc_nblk") {
            manifest.rsc_nblk = std::stoi(value);
        } else if (key == "rsc_seed_sel") {
            manifest.rsc_seed_sel = std::stoi(value);
        } else if (key == "rsc_start") {
            manifest.rsc_start = std::stoi(value);
        } else if (key == "rsc_turbo_iters") {
            manifest.rsc_turbo_iters = std::stoi(value);
        } else if (key == "rsc_sigma2_init") {
            manifest.rsc_sigma2_init = std::stod(value);
        } else if (key == "rsc_eq_sigma2_iters") {
            manifest.rsc_eq_sigma2_iters = std::stoi(value);
        } else if (key == "rsc_llr_clip") {
            manifest.rsc_llr_clip = std::stod(value);
        } else if (key == "rsc_default_order_corr_thr") {
            manifest.rsc_default_order_corr_thr = std::stod(value);
        } else if (key == "rsc_default_order_seed") {
            manifest.rsc_default_order_seed = std::stoi(value);
        }
    }

    if (manifest.ldpc_h_file.empty()) {
        throw std::runtime_error("Manifest is missing ldpc_h_file");
    }
    if (manifest.raw_packets_file.empty() || manifest.raw_cw_true_file.empty()) {
        throw std::runtime_error("Manifest is missing RAW bundle files");
    }
    if (manifest.rsc_y_file.empty() || manifest.rsc_u64_file.empty() || manifest.rsc_b128_file.empty()) {
        throw std::runtime_error("Manifest is missing RSC bundle files");
    }
    if (manifest.pilot_fracs.empty()) {
        throw std::runtime_error("Manifest is missing pilot_fracs");
    }
    return manifest;
}

BundleData load_bundle(const std::string& manifest_or_dir) {
    std::string manifest_path = manifest_or_dir;
    if (manifest_or_dir.size() < 4 ||
        manifest_or_dir.substr(manifest_or_dir.size() - 4) != ".txt") {
        manifest_path = join_path(manifest_or_dir, "manifest.txt");
    }

    BundleData bundle;
    bundle.manifest = read_manifest(manifest_path);
    const std::string& root = bundle.manifest.root_dir;

    bundle.raw_packets = read_complex_matrix(join_path(root, bundle.manifest.raw_packets_file));
    bundle.raw_cw_true = read_int_matrix(join_path(root, bundle.manifest.raw_cw_true_file));
    bundle.raw_frame_ids = read_int_vector(join_path(root, bundle.manifest.raw_frame_ids_file));
    bundle.raw_block_ids = read_int_vector(join_path(root, bundle.manifest.raw_block_ids_file));
    bundle.raw_bestd = read_int_vector(join_path(root, bundle.manifest.raw_bestd_file));

    bundle.rsc_y = read_complex_matrix(join_path(root, bundle.manifest.rsc_y_file));
    bundle.rsc_u64 = read_int_matrix(join_path(root, bundle.manifest.rsc_u64_file));
    bundle.rsc_b128 = read_int_matrix(join_path(root, bundle.manifest.rsc_b128_file));
    bundle.rsc_h = read_complex_matrix(join_path(root, bundle.manifest.rsc_h_file));
    bundle.rsc_corr = read_double_vector(join_path(root, bundle.manifest.rsc_corr_file));
    bundle.rsc_frame_ids = read_int_vector(join_path(root, bundle.manifest.rsc_frame_ids_file));
    if (!bundle.manifest.rsc_default_order_file.empty()) {
        bundle.rsc_default_order = read_int_vector(join_path(root, bundle.manifest.rsc_default_order_file));
    }

    return bundle;
}

std::vector<cd> row_as_complex(const ComplexMatrix& mat, int row, int max_cols) {
    int cols = max_cols < 0 ? static_cast<int>(mat.cols) : std::min<int>(max_cols, static_cast<int>(mat.cols));
    std::vector<cd> out(static_cast<std::size_t>(cols));
    for (int col = 0; col < cols; ++col) {
        out[static_cast<std::size_t>(col)] = mat(row, col);
    }
    return out;
}

std::vector<int> row_as_int(const IntMatrix& mat, int row, int max_cols) {
    int cols = max_cols < 0 ? static_cast<int>(mat.cols) : std::min<int>(max_cols, static_cast<int>(mat.cols));
    std::vector<int> out(static_cast<std::size_t>(cols));
    for (int col = 0; col < cols; ++col) {
        out[static_cast<std::size_t>(col)] = mat(row, col);
    }
    return out;
}

}  // namespace replayswap_cpp
