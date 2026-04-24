#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static const ImVec4 COL_BG       = ImVec4(0.039f, 0.086f, 0.157f, 1.0f);
static const ImVec4 COL_PANEL    = ImVec4(0.075f, 0.133f, 0.220f, 1.0f);
static const ImVec4 COL_ACCENT   = ImVec4(0.000f, 0.831f, 1.000f, 1.0f);
static const ImVec4 COL_ACCENT2  = ImVec4(0.000f, 1.000f, 0.533f, 1.0f);
static const ImVec4 COL_WARNING  = ImVec4(1.000f, 0.667f, 0.000f, 1.0f);
static const ImVec4 COL_TEXT     = ImVec4(0.878f, 0.910f, 0.941f, 1.0f);
static const ImVec4 COL_TEXT_DIM = ImVec4(0.439f, 0.565f, 0.690f, 1.0f);

static constexpr int FIXED_N = 2048;
static constexpr int FIXED_K = 1024;
static constexpr int FIXED_CHECKS_PER_NODE = 3;
static constexpr int FIXED_PRPRP_ITERS = 200;
static constexpr int DEFAULT_BATCH_BLOCKS = 20;
static constexpr int DFEC_PILOT_OPTION_COUNT = 6;
static const double DFEC_PILOT_OPTIONS[DFEC_PILOT_OPTION_COUNT] = {0.26, 0.31, 0.36, 0.41, 0.46, 0.51};

struct SweepResult {
    double snr_db = 0.0;
    int total_blocks = 0;
    int primary_bit_errors = 0;
    double primary_ber = 0.0;
    bool primary_hit_target = false;
    int secondary_bit_errors = 0;
    double secondary_ber = 0.0;
    bool secondary_hit_target = false;
    bool has_secondary = false;
    std::string note;
};

struct PipelineStatus {
    std::atomic<bool> running{false};
    std::atomic<int> exit_code{-1};
    bool has_secondary_curve = false;
    std::string primary_curve_label = "prprp";
    std::string secondary_curve_label;
    std::string message;
    std::vector<SweepResult> results;
    std::mutex mutex;
};

struct UiConfig {
    char channel_type[16] = "awgn";
    int min_bit_errors = 200;
    int max_packets = 20000;
    float snr_start_db = -2.0f;
    float snr_stop_db = 2.0f;
    float snr_step_db = 1.0f;
    float isi_taps[3] = {1.0f, 0.5f, 0.25f};
    char dfec_bundle_dir[512] = "DFEC_experiment/export/cpp_bundle";
    bool dfec_pilot_selected[DFEC_PILOT_OPTION_COUNT] = {false, false, true, true, true, false};
};

struct DfecPacketResult {
    double pilot_frac = 0.0;
    int packet_index = 0;
    int total_packets = 0;
    int frame = 0;
    int block = 0;
    double dfec_ber = 0.0;
    double dfe_ber = 0.0;
    double spa_ber = 0.0;
    double min_ber = 0.0;
    bool success = false;
    bool joint_valid = false;
};

struct DfecPilotProgress {
    double pilot_frac = 0.0;
    int pilot_count = 0;
    int total_packets = 0;
    int processed_packets = 0;
    int success_packets = 0;
    double success_rate = 0.0;
    double mean_dfec_ber = 0.0;
    double mean_dfe_ber = 0.0;
};

struct DfecDecodeStatus {
    std::atomic<bool> running{false};
    std::atomic<int> exit_code{-1};
    std::string message = "Ready";
    std::string bundle_dir;
    std::string output_csv;
    std::string raw_output;
    int total_packets_per_pilot = 0;
    std::vector<DfecPacketResult> packets;
    std::vector<DfecPilotProgress> pilot_progress;
    std::mutex mutex;
};


struct VerifyStats {
    int blocks = 0;
    double ber = 0.0;
};

static PipelineStatus pipeline_status;
static DfecDecodeStatus dfec_status;
static std::mutex cout_mutex;
static std::thread background_thread;
static std::thread dfec_thread;
static UiConfig config;
static std::ofstream log_file;
static std::mutex dropped_path_mutex;
static std::string dropped_path;

static bool compare_isi_receivers(const UiConfig& cfg) {
    return std::strcmp(cfg.channel_type, "isi") == 0;
}


static void safe_print(const std::string& line) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << line << std::endl;
    if (log_file.is_open()) {
        log_file << line << std::endl;
        log_file.flush();
    }
}

static void update_pipeline_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(pipeline_status.mutex);
    pipeline_status.message = message;
}

static void update_partial_result(const SweepResult& result) {
    std::lock_guard<std::mutex> lock(pipeline_status.mutex);
    if (pipeline_status.results.empty()
        || std::fabs(pipeline_status.results.back().snr_db - result.snr_db) > 1e-9) {
        pipeline_status.results.push_back(result);
    } else {
        pipeline_status.results.back() = result;
    }
}

static void update_dfec_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(dfec_status.mutex);
    dfec_status.message = message;
}

static std::string trim_copy(const std::string& text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }

    return text.substr(begin, end - begin);
}

static std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

static std::map<std::string, std::string> parse_key_value_fields(const std::string& line, const std::string& prefix) {
    std::map<std::string, std::string> fields;
    if (line.rfind(prefix, 0) != 0) {
        return fields;
    }

    std::istringstream iss(line.substr(prefix.size()));
    std::string token;
    while (iss >> token) {
        size_t eq = token.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= token.size()) {
            continue;
        }
        fields[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return fields;
}

static DfecPilotProgress* find_dfec_pilot_progress_locked(double pilot_frac) {
    for (DfecPilotProgress& progress : dfec_status.pilot_progress) {
        if (std::fabs(progress.pilot_frac - pilot_frac) <= 1e-9) {
            return &progress;
        }
    }
    dfec_status.pilot_progress.push_back(DfecPilotProgress());
    DfecPilotProgress& progress = dfec_status.pilot_progress.back();
    progress.pilot_frac = pilot_frac;
    std::sort(dfec_status.pilot_progress.begin(), dfec_status.pilot_progress.end(),
              [](const DfecPilotProgress& a, const DfecPilotProgress& b) {
                  return a.pilot_frac < b.pilot_frac;
              });
    for (DfecPilotProgress& item : dfec_status.pilot_progress) {
        if (std::fabs(item.pilot_frac - pilot_frac) <= 1e-9) {
            return &item;
        }
    }
    return nullptr;
}

static void append_dfec_output_line_locked(const std::string& line) {
    dfec_status.raw_output += line;
    dfec_status.raw_output += '\n';
    static constexpr size_t MAX_LOG_CHARS = 250000;
    if (dfec_status.raw_output.size() > MAX_LOG_CHARS) {
        dfec_status.raw_output.erase(0, dfec_status.raw_output.size() - MAX_LOG_CHARS);
    }
}

static void handle_dfec_output_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(dfec_status.mutex);
    append_dfec_output_line_locked(line);

    if (line.rfind("UI_RUN ", 0) == 0) {
        std::map<std::string, std::string> fields = parse_key_value_fields(line, "UI_RUN ");
        auto it = fields.find("packets");
        if (it != fields.end()) {
            dfec_status.total_packets_per_pilot = std::max(0, std::atoi(it->second.c_str()));
        }
        return;
    }

    if (line.rfind("UI_PILOT ", 0) == 0) {
        std::map<std::string, std::string> fields = parse_key_value_fields(line, "UI_PILOT ");
        auto pilot_it = fields.find("pilot");
        if (pilot_it == fields.end()) {
            return;
        }
        double pilot_frac = std::atof(pilot_it->second.c_str());
        DfecPilotProgress* progress = find_dfec_pilot_progress_locked(pilot_frac);
        if (!progress) {
            return;
        }
        auto pilot_count_it = fields.find("pilot_count");
        if (pilot_count_it != fields.end()) {
            progress->pilot_count = std::max(0, std::atoi(pilot_count_it->second.c_str()));
        }
        auto total_packets_it = fields.find("total_packets");
        if (total_packets_it != fields.end()) {
            progress->total_packets = std::max(0, std::atoi(total_packets_it->second.c_str()));
        }
        return;
    }

    if (line.rfind("UI_PILOT_DONE ", 0) == 0) {
        std::map<std::string, std::string> fields = parse_key_value_fields(line, "UI_PILOT_DONE ");
        auto pilot_it = fields.find("pilot");
        if (pilot_it == fields.end()) {
            return;
        }
        double pilot_frac = std::atof(pilot_it->second.c_str());
        DfecPilotProgress* progress = find_dfec_pilot_progress_locked(pilot_frac);
        if (!progress) {
            return;
        }
        auto processed_it = fields.find("processed_packets");
        auto success_it = fields.find("success_packets");
        auto success_rate_it = fields.find("success_rate");
        auto mean_dfec_it = fields.find("mean_dfec_ber");
        auto mean_dfe_it = fields.find("mean_dfe_ber");
        if (processed_it != fields.end()) progress->processed_packets = std::max(0, std::atoi(processed_it->second.c_str()));
        if (success_it != fields.end()) progress->success_packets = std::max(0, std::atoi(success_it->second.c_str()));
        if (success_rate_it != fields.end()) progress->success_rate = std::atof(success_rate_it->second.c_str());
        if (mean_dfec_it != fields.end()) progress->mean_dfec_ber = std::atof(mean_dfec_it->second.c_str());
        if (mean_dfe_it != fields.end()) progress->mean_dfe_ber = std::atof(mean_dfe_it->second.c_str());
        return;
    }

    if (line.rfind("UI_PACKET ", 0) == 0) {
        std::map<std::string, std::string> fields = parse_key_value_fields(line, "UI_PACKET ");
        auto pilot_it = fields.find("pilot");
        if (pilot_it == fields.end()) {
            return;
        }

        DfecPacketResult packet;
        packet.pilot_frac = std::atof(pilot_it->second.c_str());
        packet.packet_index = std::max(0, std::atoi(fields["packet_index"].c_str()));
        packet.total_packets = std::max(0, std::atoi(fields["total_packets"].c_str()));
        packet.frame = std::max(0, std::atoi(fields["frame"].c_str()));
        packet.block = std::max(0, std::atoi(fields["block"].c_str()));
        packet.dfec_ber = std::atof(fields["dfec_ber"].c_str());
        packet.dfe_ber = std::atof(fields["dfe_ber"].c_str());
        packet.spa_ber = std::atof(fields["spa_ber"].c_str());
        packet.min_ber = std::atof(fields["min_ber"].c_str());
        packet.success = std::atoi(fields["success"].c_str()) != 0;
        packet.joint_valid = std::atoi(fields["joint_valid"].c_str()) != 0;
        dfec_status.packets.push_back(packet);

        DfecPilotProgress* progress = find_dfec_pilot_progress_locked(packet.pilot_frac);
        if (!progress) {
            return;
        }
        progress->total_packets = std::max(progress->total_packets, packet.total_packets);
        progress->processed_packets = std::max(progress->processed_packets, packet.packet_index);
        if (packet.success) {
            progress->success_packets += 1;
        }
        const int count = std::max(1, progress->processed_packets);
        progress->mean_dfec_ber += (packet.dfec_ber - progress->mean_dfec_ber) / static_cast<double>(count);
        progress->mean_dfe_ber += (packet.dfe_ber - progress->mean_dfe_ber) / static_cast<double>(count);
        progress->success_rate = progress->processed_packets > 0
            ? static_cast<double>(progress->success_packets) / static_cast<double>(progress->processed_packets)
            : 0.0;
        return;
    }

    if (line.rfind("UI_DONE ", 0) == 0) {
        std::map<std::string, std::string> fields = parse_key_value_fields(line, "UI_DONE ");
        auto csv_it = fields.find("output_csv");
        if (csv_it != fields.end()) {
            dfec_status.output_csv = csv_it->second;
        }
    }
}

static std::string fmt_double(double value, int precision = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

static double compute_sigma(double ebn0_db) {
    double rate = static_cast<double>(FIXED_K) / static_cast<double>(FIXED_N);
    double ebn0 = std::pow(10.0, ebn0_db / 10.0);
    return std::sqrt(1.0 / (2.0 * rate * ebn0));
}

static double compute_dfe_sigma(const UiConfig& cfg, double sigma) {
    if (std::strcmp(cfg.channel_type, "isi") != 0) {
        return sigma;
    }
    return sigma / std::fabs(cfg.isi_taps[0]);
}

static bool file_exists(const char* path) {
    FILE* fp = std::fopen(path, "r");
    if (!fp) {
        return false;
    }
    std::fclose(fp);
    return true;
}

static std::string normalize_bundle_path(const std::string& raw_path) {
    if (raw_path.empty()) {
        return "";
    }

    try {
        std::filesystem::path path = std::filesystem::absolute(raw_path);
        if (std::filesystem::is_directory(path)) {
            if (std::filesystem::exists(path / "manifest.txt")) {
                return path.string();
            }
            return path.string();
        }

        std::filesystem::path parent = path.parent_path();
        if (path.filename() == "manifest.txt") {
            return parent.string();
        }
        if (!parent.empty() && std::filesystem::exists(parent / "manifest.txt")) {
            return parent.string();
        }
        return raw_path;
    } catch (const std::exception&) {
        return raw_path;
    }
}

static void set_dfec_bundle_dir(const std::string& path) {
    std::string normalized = normalize_bundle_path(trim_copy(path));
    std::snprintf(config.dfec_bundle_dir, sizeof(config.dfec_bundle_dir), "%s", normalized.c_str());
}

static void glfw_file_drop_callback(GLFWwindow*, int count, const char** paths) {
    if (count <= 0 || paths == nullptr || paths[0] == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(dropped_path_mutex);
    dropped_path = paths[0];
}

static void apply_pending_dropped_path() {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(dropped_path_mutex);
        path.swap(dropped_path);
    }
    if (!path.empty()) {
        set_dfec_bundle_dir(path);
    }
}

static void cleanup_file(const std::string& path) {
    if (std::remove(path.c_str()) != 0 && errno != ENOENT) {
        safe_print("Warning: could not remove " + path);
    }
}

static bool run_command_capture(const std::string& cmd, std::string* output) {
    safe_print("$ " + cmd);

    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        if (output) {
            *output = "Failed to start command";
        }
        return false;
    }

    char buffer[512];
    std::string collected;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        collected += buffer;
    }

    int rc = pclose(pipe);
    if (output) {
        *output = collected;
    }

    if (!collected.empty()) {
        std::istringstream lines(collected);
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty()) {
                safe_print(line);
            }
        }
    }

    return rc == 0;
}

static bool run_command_stream(const std::string& cmd,
                               const std::function<void(const std::string&)>& on_line,
                               std::string* output) {
    safe_print("$ " + cmd);

    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        if (output) {
            *output = "Failed to start command";
        }
        return false;
    }

    char buffer[512];
    std::string collected;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line = buffer;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        collected += buffer;
        if (!line.empty()) {
            safe_print(line);
            if (on_line) {
                on_line(line);
            }
        }
    }

    int rc = pclose(pipe);
    if (output) {
        *output = collected;
    }
    return rc == 0;
}

static bool run_command(const std::string& cmd) {
    std::string output;
    return run_command_capture(cmd, &output);
}

static bool parse_verify_output(const std::string& output, VerifyStats* stats) {
    std::smatch match;
    std::regex block_re(R"(Block counts:\s+tot\s+(\d+),)");
    std::regex ber_re(R"(Bit error rate \(on message bits only\):\s+([0-9eE+\-.]+))");

    if (!std::regex_search(output, match, block_re)) {
        return false;
    }
    stats->blocks = std::stoi(match[1].str());

    if (!std::regex_search(output, match, ber_re)) {
        return false;
    }
    stats->ber = std::stod(match[1].str());
    return true;
}

static std::vector<double> make_snr_sweep(double start_db, double stop_db, double step_db) {
    std::vector<double> sweep;
    for (double snr = start_db; snr <= stop_db + step_db * 0.5; snr += step_db) {
        sweep.push_back(snr);
    }
    return sweep;
}

static std::vector<double> selected_dfec_pilots(const UiConfig& cfg) {
    std::vector<double> pilots;
    for (int i = 0; i < DFEC_PILOT_OPTION_COUNT; ++i) {
        if (cfg.dfec_pilot_selected[i]) {
            pilots.push_back(DFEC_PILOT_OPTIONS[i]);
        }
    }
    return pilots;
}

static void normalize_isi_taps() {
    float power = config.isi_taps[0] * config.isi_taps[0] +
                  config.isi_taps[1] * config.isi_taps[1] +
                  config.isi_taps[2] * config.isi_taps[2];

    if (power > 1e-6f) {
        float scale = 1.0f / std::sqrt(power);
        config.isi_taps[0] *= scale;
        config.isi_taps[1] *= scale;
        config.isi_taps[2] *= scale;
    }
}

static std::string build_channel_command(const UiConfig& cfg,
                                         const std::string& enc_file,
                                         const std::string& rec_file,
                                         double sigma) {
    std::ostringstream cmd;
    cmd << std::scientific << std::setprecision(12);
    if (std::strcmp(cfg.channel_type, "isi") == 0) {
        cmd << "./transmit_isi " << enc_file << " " << rec_file << " 1 " << sigma
            << " " << cfg.isi_taps[0]
            << " " << cfg.isi_taps[1]
            << " " << cfg.isi_taps[2];
    } else {
        cmd << "./transmit " << enc_file << " " << rec_file << " 1 awgn " << sigma;
    }
    return cmd.str();
}

static bool run_sweep(const UiConfig& cfg) {
    pipeline_status.running = true;
    pipeline_status.exit_code = -1;

    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    std::ostringstream log_path;
    log_path << cwd << "/bp_decoder_sweep.log";

    log_file.open(log_path.str(), std::ios::app);
    safe_print("=== BP Decoder Sweep Started ===");
    safe_print("Working directory: " + std::string(cwd));

    {
        std::lock_guard<std::mutex> lock(pipeline_status.mutex);
        pipeline_status.results.clear();
        pipeline_status.has_secondary_curve = compare_isi_receivers(cfg);
        pipeline_status.primary_curve_label = compare_isi_receivers(cfg)
            ? "joint dfec + prprp"
            : "prprp";
        pipeline_status.secondary_curve_label = compare_isi_receivers(cfg)
            ? "dfe + prprp"
            : "";
        pipeline_status.message = "Preparing backend tools";
    }

    if (!file_exists("Makefile")) {
        std::ostringstream msg;
        msg << "Error: Makefile not found. Log: " << log_path.str();
        update_pipeline_message(msg.str());
        safe_print("FAILED: Makefile not found");
        pipeline_status.exit_code = 1;
        pipeline_status.running = false;
        if (log_file.is_open()) log_file.close();
        return false;
    }

    if (!run_command("make progs")) {
        update_pipeline_message("Failed to build backend tools with make progs");
        pipeline_status.exit_code = 2;
        pipeline_status.running = false;
        return false;
    }

    const std::string run_base = "bp_gui_fixed";
    const std::string pchk_file = run_base + ".pchk";
    const std::string gen_file = run_base + ".gen";

    {
        std::ostringstream cmd;
        cmd << "./make-ldpc " << pchk_file << " " << (FIXED_N - FIXED_K) << " " << FIXED_N
            << " 1 evenboth " << FIXED_CHECKS_PER_NODE << " no4cycle";
        if (!run_command(cmd.str())) {
            update_pipeline_message("make-ldpc failed");
            pipeline_status.exit_code = 3;
            pipeline_status.running = false;
            return false;
        }
    }

    {
        std::ostringstream cmd;
        cmd << "./make-gen " << pchk_file << " " << gen_file << " sparse";
        if (!run_command(cmd.str())) {
            update_pipeline_message("make-gen failed");
            pipeline_status.exit_code = 4;
            pipeline_status.running = false;
            if (log_file.is_open()) log_file.close();
            return false;
        }
    }

    std::vector<double> sweep = make_snr_sweep(cfg.snr_start_db, cfg.snr_stop_db, cfg.snr_step_db);

    for (size_t snr_idx = 0; snr_idx < sweep.size(); ++snr_idx) {
        double snr_db = sweep[snr_idx];
        double sigma = compute_sigma(snr_db);
        SweepResult result;
        result.snr_db = snr_db;
        result.has_secondary = compare_isi_receivers(cfg);

        for (int batch_idx = 0; result.total_blocks < cfg.max_packets; ++batch_idx) {
            int batch_blocks = std::min(DEFAULT_BATCH_BLOCKS, cfg.max_packets - result.total_blocks);
            if (batch_blocks <= 0) {
                break;
            }

            std::ostringstream prefix_builder;
            prefix_builder << run_base << "_snr_" << snr_idx << "_batch_" << batch_idx;
            std::string prefix = prefix_builder.str();
            std::string src_file = prefix + ".src";
            std::string rec_file = prefix + ".rec";
            std::string eq_file = prefix + ".eq";
            std::string joint_dec_file = prefix + ".dec-joint";
            std::string dfe_dec_file = prefix + ".dec-prprp";
            std::string awgn_dec_file = prefix + ".dec-prprp";

            std::ostringstream src_cmd;
            src_cmd << "./rand-src " << src_file << " 1 " << FIXED_K << "x" << batch_blocks;
            if (!run_command(src_cmd.str())) {
                safe_print("FAILED: rand-src at SNR " + fmt_double(snr_db, 2) + " dB");
                pipeline_status.exit_code = 5;
                pipeline_status.running = false;
                cleanup_file(src_file);
                cleanup_file(rec_file);
                cleanup_file(eq_file);
                cleanup_file(joint_dec_file);
                cleanup_file(dfe_dec_file);
                cleanup_file(awgn_dec_file);
                if (log_file.is_open()) log_file.close();
                return false;
            }

            std::string enc_file = prefix + ".enc";
            std::ostringstream enc_cmd;
            enc_cmd << "./encode " << pchk_file << " " << gen_file << " " << src_file << " " << enc_file;
            if (!run_command(enc_cmd.str())) {
                safe_print("FAILED: encode at SNR " + fmt_double(snr_db, 2) + " dB");
                pipeline_status.exit_code = 5;
                pipeline_status.running = false;
                cleanup_file(src_file);
                cleanup_file(enc_file);
                cleanup_file(rec_file);
                cleanup_file(eq_file);
                cleanup_file(joint_dec_file);
                cleanup_file(dfe_dec_file);
                cleanup_file(awgn_dec_file);
                if (log_file.is_open()) log_file.close();
                return false;
            }

            std::ostringstream msg;
            msg << "SNR " << fmt_double(snr_db, 2) << " dB: ";
            if (compare_isi_receivers(cfg)) {
                msg << "joint " << result.primary_bit_errors << "/" << cfg.min_bit_errors
                    << ", dfe " << result.secondary_bit_errors << "/" << cfg.min_bit_errors
                    << " after " << result.total_blocks << "/" << cfg.max_packets
                    << " packets";
            } else {
                msg << result.primary_bit_errors << "/" << cfg.min_bit_errors
                    << " bit errors after " << result.total_blocks << "/" << cfg.max_packets
                    << " packets";
            }
            update_pipeline_message(msg.str());

            if (!run_command(build_channel_command(cfg, enc_file, rec_file, sigma))) {
                char cwd2[1024];
                getcwd(cwd2, sizeof(cwd2));
                std::ostringstream msg;
                msg << "Channel simulation failed at SNR " << fmt_double(snr_db, 2) << " dB. See log: " << cwd2 << "/bp_decoder_sweep.log";
                update_pipeline_message(msg.str());
                safe_print("FAILED: Channel simulation at SNR " + fmt_double(snr_db, 2) + " dB");
                pipeline_status.exit_code = 5;
                pipeline_status.running = false;
                cleanup_file(src_file);
                cleanup_file(enc_file);
                cleanup_file(rec_file);
                cleanup_file(eq_file);
                cleanup_file(joint_dec_file);
                cleanup_file(dfe_dec_file);
                cleanup_file(awgn_dec_file);
                if (log_file.is_open()) log_file.close();
                return false;
            }

            VerifyStats primary_stats;
            VerifyStats secondary_stats;
            if (compare_isi_receivers(cfg)) {
                std::string decode_output;
                std::ostringstream joint_cmd;
                joint_cmd << std::scientific << std::setprecision(12)
                          << "./joint_dfec_bp " << pchk_file << " " << rec_file << " " << joint_dec_file
                          << " " << cfg.isi_taps[0]
                          << " " << cfg.isi_taps[1]
                          << " " << cfg.isi_taps[2];
                if (!run_command_capture(joint_cmd.str(), &decode_output)) {
                    update_pipeline_message("Joint DFEC-BP failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 7;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }

                std::string verify_output;
                std::ostringstream verify_cmd;
                verify_cmd << "./verify " << pchk_file << " " << joint_dec_file << " "
                           << gen_file << " " << src_file;
                if (!run_command_capture(verify_cmd.str(), &verify_output)
                    || !parse_verify_output(verify_output, &primary_stats)) {
                    update_pipeline_message("Joint verify failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 8;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }

                std::string decode_input_file = rec_file;
                double decode_sigma = sigma;

                if (std::strcmp(cfg.channel_type, "isi") == 0) {
                    std::ostringstream dfe_cmd;
                    dfe_cmd << "./equalize_dfe " << rec_file << " " << eq_file << " " << FIXED_N
                            << " " << cfg.isi_taps[0]
                            << " " << cfg.isi_taps[1]
                            << " " << cfg.isi_taps[2];
                    if (!run_command(dfe_cmd.str())) {
                        update_pipeline_message("DFE failed at SNR " + fmt_double(snr_db, 2) + " dB");
                        pipeline_status.exit_code = 6;
                        pipeline_status.running = false;
                        cleanup_file(src_file);
                        cleanup_file(enc_file);
                        cleanup_file(rec_file);
                        cleanup_file(eq_file);
                        cleanup_file(joint_dec_file);
                        cleanup_file(dfe_dec_file);
                        cleanup_file(awgn_dec_file);
                        if (log_file.is_open()) log_file.close();
                        return false;
                    }
                    decode_input_file = eq_file;
                    decode_sigma = compute_dfe_sigma(cfg, sigma);
                }

                decode_output.clear();
                std::ostringstream decode_cmd;
                decode_cmd << std::scientific << std::setprecision(12)
                           << "./decode " << pchk_file << " " << decode_input_file << " " << dfe_dec_file
                           << " awgn " << decode_sigma << " prprp " << FIXED_PRPRP_ITERS;
                if (!run_command_capture(decode_cmd.str(), &decode_output)) {
                    update_pipeline_message("DFE + PRPRP failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 7;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }

                verify_output.clear();
                verify_cmd.str("");
                verify_cmd.clear();
                verify_cmd << "./verify " << pchk_file << " " << dfe_dec_file << " "
                           << gen_file << " " << src_file;
                if (!run_command_capture(verify_cmd.str(), &verify_output)
                    || !parse_verify_output(verify_output, &secondary_stats)) {
                    update_pipeline_message("DFE verify failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 8;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }
            } else {
                std::string decode_output;
                std::ostringstream decode_cmd;
                decode_cmd << std::scientific << std::setprecision(12)
                           << "./decode " << pchk_file << " " << rec_file << " " << awgn_dec_file
                           << " awgn " << sigma << " prprp " << FIXED_PRPRP_ITERS;
                if (!run_command_capture(decode_cmd.str(), &decode_output)) {
                    update_pipeline_message("PRPRP decode failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 7;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }

                std::string verify_output;
                std::ostringstream verify_cmd;
                verify_cmd << "./verify " << pchk_file << " " << awgn_dec_file << " "
                           << gen_file << " " << src_file;
                if (!run_command_capture(verify_cmd.str(), &verify_output)
                    || !parse_verify_output(verify_output, &primary_stats)) {
                    update_pipeline_message("Verify failed at SNR " + fmt_double(snr_db, 2) + " dB");
                    pipeline_status.exit_code = 8;
                    pipeline_status.running = false;
                    cleanup_file(src_file);
                    cleanup_file(enc_file);
                    cleanup_file(rec_file);
                    cleanup_file(eq_file);
                    cleanup_file(joint_dec_file);
                    cleanup_file(dfe_dec_file);
                    cleanup_file(awgn_dec_file);
                    if (log_file.is_open()) log_file.close();
                    return false;
                }
            }

            cleanup_file(src_file);
            cleanup_file(enc_file);
            cleanup_file(rec_file);
            cleanup_file(eq_file);
            cleanup_file(joint_dec_file);
            cleanup_file(dfe_dec_file);
            cleanup_file(awgn_dec_file);

            int used_blocks = compare_isi_receivers(cfg)
                ? std::min(primary_stats.blocks, secondary_stats.blocks)
                : primary_stats.blocks;
            if (used_blocks <= 0) {
                update_pipeline_message("No decoded blocks were verified at SNR " + fmt_double(snr_db, 2) + " dB");
                pipeline_status.exit_code = 8;
                pipeline_status.running = false;
                if (log_file.is_open()) log_file.close();
                return false;
            }
            int primary_batch_errors = static_cast<int>(std::llround(primary_stats.ber * used_blocks * FIXED_K));
            result.total_blocks += used_blocks;
            result.primary_bit_errors += primary_batch_errors;
            result.primary_ber = result.total_blocks > 0
                ? static_cast<double>(result.primary_bit_errors) / static_cast<double>(result.total_blocks * FIXED_K)
                : 0.0;

            if (compare_isi_receivers(cfg)) {
                int secondary_batch_errors = static_cast<int>(std::llround(secondary_stats.ber * used_blocks * FIXED_K));
                result.secondary_bit_errors += secondary_batch_errors;
                result.secondary_ber = result.total_blocks > 0
                    ? static_cast<double>(result.secondary_bit_errors) / static_cast<double>(result.total_blocks * FIXED_K)
                    : 0.0;
                result.note = (primary_stats.blocks < batch_blocks || secondary_stats.blocks < batch_blocks)
                    ? "short verify output"
                    : "";
            } else {
                result.note = primary_stats.blocks < batch_blocks ? "short verify output" : "";
            }

            update_partial_result(result);

            result.primary_hit_target = result.primary_bit_errors >= cfg.min_bit_errors;
            result.secondary_hit_target = compare_isi_receivers(cfg)
                ? result.secondary_bit_errors >= cfg.min_bit_errors
                : false;

            bool done = compare_isi_receivers(cfg)
                ? (result.primary_hit_target && result.secondary_hit_target)
                : result.primary_hit_target;

            if (done) {
                result.note = compare_isi_receivers(cfg)
                    ? "both targets reached"
                    : "min errors reached";
                update_partial_result(result);
                break;
            }
        }

        if (((compare_isi_receivers(cfg) && !(result.primary_hit_target && result.secondary_hit_target))
             || (!compare_isi_receivers(cfg) && !result.primary_hit_target))
            && result.total_blocks >= cfg.max_packets) {
            result.note = "max packets reached";
            update_partial_result(result);
        }
    }

    cleanup_file(pchk_file);
    cleanup_file(gen_file);

    char cwd2[1024];
    getcwd(cwd2, sizeof(cwd2));
    std::ostringstream final_msg;
    final_msg << "Sweep completed - Log: " << cwd2 << "/bp_decoder_sweep.log";
    update_pipeline_message(final_msg.str());
    safe_print("=== Sweep Completed Successfully ===");
    pipeline_status.exit_code = 0;
    pipeline_status.running = false;
    if (log_file.is_open()) log_file.close();
    return true;
}

static void start_background_pipeline() {
    if (pipeline_status.running) {
        safe_print("Sweep already running.");
        return;
    }
    if (background_thread.joinable()) {
        background_thread.join();
    }
    background_thread = std::thread([]() {
        run_sweep(config);
    });
}

static bool run_dfec_dataset_decode(const UiConfig& cfg) {
    dfec_status.running = true;
    dfec_status.exit_code = -1;

    {
        std::lock_guard<std::mutex> lock(dfec_status.mutex);
        dfec_status.message = "Preparing C++ DFEC decoder";
        dfec_status.bundle_dir = normalize_bundle_path(cfg.dfec_bundle_dir);
        dfec_status.output_csv.clear();
        dfec_status.raw_output.clear();
        dfec_status.total_packets_per_pilot = 0;
        dfec_status.packets.clear();
        dfec_status.pilot_progress.clear();
    }

    std::string bundle_dir = normalize_bundle_path(cfg.dfec_bundle_dir);
    if (bundle_dir.empty()) {
        update_dfec_message("Choose a DFEC bundle directory first");
        dfec_status.exit_code = 21;
        dfec_status.running = false;
        return false;
    }

    std::filesystem::path manifest_path = std::filesystem::path(bundle_dir) / "manifest.txt";
    if (!file_exists(manifest_path.string().c_str())) {
        update_dfec_message("Bundle is missing manifest.txt");
        dfec_status.exit_code = 22;
        dfec_status.running = false;
        return false;
    }

    std::vector<double> pilots = selected_dfec_pilots(cfg);
    if (pilots.empty()) {
        update_dfec_message("Select at least one pilot ratio");
        dfec_status.exit_code = 23;
        dfec_status.running = false;
        return false;
    }

    update_dfec_message("Building C++ DFEC runner");
    if (!run_command("make -C DFEC_experiment/cpp")) {
        update_dfec_message("Failed to build DFEC_experiment/cpp");
        dfec_status.exit_code = 24;
        dfec_status.running = false;
        return false;
    }

    std::filesystem::path output_csv = std::filesystem::path("/tmp") / "ldpc_gui_dfec_results.csv";
    std::ostringstream pilot_list;
    for (size_t i = 0; i < pilots.size(); ++i) {
        if (i != 0) {
            pilot_list << ",";
        }
        pilot_list << std::fixed << std::setprecision(2) << pilots[i];
    }

    std::ostringstream cmd;
    cmd << "./DFEC_experiment/cpp/run_experiment_cpp "
        << shell_quote(bundle_dir) << " "
        << shell_quote(output_csv.string())
        << " --pilot-fracs " << shell_quote(pilot_list.str());

    update_dfec_message("Decoding packets one by one");
    bool ok = run_command_stream(cmd.str(), [](const std::string& line) {
        handle_dfec_output_line(line);
    }, nullptr);

    if (!ok) {
        update_dfec_message("C++ DFEC decode failed");
        dfec_status.exit_code = 25;
        dfec_status.running = false;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(dfec_status.mutex);
        dfec_status.output_csv = output_csv.string();
        dfec_status.message = "Decode completed";
    }
    dfec_status.exit_code = 0;
    dfec_status.running = false;
    return true;
}

static void start_background_dfec_decode() {
    if (dfec_status.running) {
        safe_print("DFEC decode already running.");
        return;
    }
    if (dfec_thread.joinable()) {
        dfec_thread.join();
    }

    UiConfig cfg_snapshot = config;
    dfec_thread = std::thread([cfg_snapshot]() {
        run_dfec_dataset_decode(cfg_snapshot);
    });
}

static void apply_theme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 8.0f;
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.WindowPadding = ImVec2(14, 14);
    style.FramePadding = ImVec2(10, 6);
    style.ItemSpacing = ImVec2(10, 10);

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg] = COL_BG;
    c[ImGuiCol_ChildBg] = ImVec4(0.055f, 0.110f, 0.190f, 1.0f);
    c[ImGuiCol_PopupBg] = COL_PANEL;
    c[ImGuiCol_Border] = ImVec4(COL_TEXT_DIM.x, COL_TEXT_DIM.y, COL_TEXT_DIM.z, 0.5f);
    c[ImGuiCol_FrameBg] = ImVec4(0.08f, 0.16f, 0.28f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.25f, 0.42f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.15f, 0.30f, 0.50f, 1.0f);
    c[ImGuiCol_TitleBg] = COL_PANEL;
    c[ImGuiCol_TitleBgActive] = ImVec4(0.05f, 0.15f, 0.30f, 1.0f);
    c[ImGuiCol_Text] = COL_TEXT;
    c[ImGuiCol_TextDisabled] = COL_TEXT_DIM;
    c[ImGuiCol_Button] = ImVec4(0.00f, 0.45f, 0.70f, 0.7f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.00f, 0.60f, 0.90f, 0.85f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.00f, 0.75f, 1.00f, 1.0f);
    c[ImGuiCol_SliderGrab] = COL_ACCENT;
    c[ImGuiCol_SliderGrabActive] = COL_ACCENT2;
    c[ImGuiCol_Header] = ImVec4(0.1f, 0.2f, 0.4f, 0.5f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.1f, 0.3f, 0.5f, 0.7f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.1f, 0.4f, 0.6f, 0.9f);
    c[ImGuiCol_Separator] = ImVec4(COL_ACCENT.x, COL_ACCENT.y, COL_ACCENT.z, 0.3f);
}

static void draw_channel_plot() {
    ImGui::TextColored(COL_ACCENT, "ISI Channel Preview");

    ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x, 150.0f);
    if (plot_size.x < 200.0f) {
        plot_size.x = 200.0f;
    }

    ImGui::BeginChild("channel_plot", plot_size, true);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 p1 = ImVec2(p0.x + plot_size.x - 16.0f, p0.y + plot_size.y - 16.0f);
    draw_list->AddRectFilled(p0, p1, IM_COL32(16, 34, 58, 255), 6.0f);
    draw_list->AddRect(p0, p1, IM_COL32(90, 140, 180, 160), 6.0f);

    float max_mag = std::max(config.isi_taps[0], std::max(config.isi_taps[1], config.isi_taps[2]));
    max_mag = std::max(max_mag, 1e-3f);

    for (int i = 0; i < 3; ++i) {
        float x = p0.x + 40.0f + i * ((p1.x - p0.x - 80.0f) / 2.0f);
        float y_base = p1.y - 24.0f;
        float y_top = y_base - (config.isi_taps[i] / max_mag) * (plot_size.y - 56.0f);
        draw_list->AddLine(ImVec2(x, y_base), ImVec2(x, y_top), IM_COL32(0, 212, 255, 255), 4.0f);
        draw_list->AddCircleFilled(ImVec2(x, y_top), 5.0f, IM_COL32(0, 255, 136, 255));

        std::ostringstream tap_label;
        tap_label << "h[" << i << "]";
        draw_list->AddText(ImVec2(x - 10.0f, y_base + 6.0f), IM_COL32(220, 232, 240, 255), tap_label.str().c_str());

        std::ostringstream mag_label;
        mag_label << std::fixed << std::setprecision(2) << config.isi_taps[i];
        draw_list->AddText(ImVec2(x - 12.0f, y_top - 22.0f), IM_COL32(0, 255, 136, 255), mag_label.str().c_str());
    }

    ImGui::EndChild();
}

static float map_plot_x(double x, double x_min, double x_max, float plot_left, float plot_right) {
    if (std::fabs(x_max - x_min) < 1e-9) {
        return 0.5f * (plot_left + plot_right);
    }
    double t = (x - x_min) / (x_max - x_min);
    return plot_left + static_cast<float>(t) * (plot_right - plot_left);
}

static float map_plot_y(double y, double y_min, double y_max, float plot_top, float plot_bottom) {
    double ly = std::log10(std::max(y, 1e-12));
    double lmin = std::log10(y_min);
    double lmax = std::log10(y_max);
    if (std::fabs(lmax - lmin) < 1e-9) {
        return 0.5f * (plot_top + plot_bottom);
    }
    double t = (ly - lmin) / (lmax - lmin);
    return plot_bottom - static_cast<float>(t) * (plot_bottom - plot_top);
}

static void draw_dashed_line(ImDrawList* draw_list,
                             const ImVec2& a,
                             const ImVec2& b,
                             ImU32 color,
                             float thickness,
                             float dash_length = 7.0f,
                             float gap_length = 4.0f) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-3f) {
        return;
    }

    float ux = dx / len;
    float uy = dy / len;
    for (float pos = 0.0f; pos < len; pos += dash_length + gap_length) {
        float end = std::min(pos + dash_length, len);
        ImVec2 p0(a.x + ux * pos, a.y + uy * pos);
        ImVec2 p1(a.x + ux * end, a.y + uy * end);
        draw_list->AddLine(p0, p1, color, thickness);
    }
}

static void draw_marker_circle(ImDrawList* draw_list, const ImVec2& p, ImU32 color) {
    draw_list->AddCircleFilled(p, 4.5f, IM_COL32(255, 255, 255, 255), 16);
    draw_list->AddCircle(p, 4.5f, color, 16, 1.4f);
}

static void draw_marker_cross(ImDrawList* draw_list, const ImVec2& p, ImU32 color) {
    draw_list->AddLine(ImVec2(p.x - 4.0f, p.y - 4.0f), ImVec2(p.x + 4.0f, p.y + 4.0f), color, 1.4f);
    draw_list->AddLine(ImVec2(p.x - 4.0f, p.y + 4.0f), ImVec2(p.x + 4.0f, p.y - 4.0f), color, 1.4f);
}

static void draw_ieee_plot_legend(ImDrawList* draw_list,
                                   float x,
                                   float y,
                                   const std::string& primary_label,
                                   const std::string& secondary_label,
                                   bool has_secondary) {
    const ImU32 primary_color = IM_COL32(16, 16, 16, 255);
    const ImU32 secondary_color = IM_COL32(120, 120, 120, 255);
    const ImU32 text_color = IM_COL32(24, 24, 24, 255);

    draw_list->AddLine(ImVec2(x, y), ImVec2(x + 26.0f, y), primary_color, 1.3f);
    draw_marker_circle(draw_list, ImVec2(x + 13.0f, y), primary_color);
    draw_list->AddText(ImVec2(x + 34.0f, y - 7.0f), text_color, primary_label.c_str());

    if (has_secondary) {
        float y2 = y + 18.0f;
        draw_dashed_line(draw_list, ImVec2(x, y2), ImVec2(x + 26.0f, y2), secondary_color, 1.3f);
        draw_marker_cross(draw_list, ImVec2(x + 13.0f, y2), secondary_color);
        draw_list->AddText(ImVec2(x + 34.0f, y2 - 7.0f), text_color, secondary_label.c_str());
    }
}

static void draw_ieee_ber_plot(const std::vector<SweepResult>& results,
                               const std::string& primary_label,
                               const std::string& secondary_label,
                               bool has_secondary) {
    const ImU32 canvas_bg = IM_COL32(252, 252, 252, 255);
    const ImU32 axis_color = IM_COL32(16, 16, 16, 255);
    const ImU32 text_color = IM_COL32(24, 24, 24, 255);
    const ImU32 major_grid = IM_COL32(178, 178, 178, 110);
    const ImU32 minor_grid = IM_COL32(204, 204, 204, 60);
    const ImU32 primary_color = IM_COL32(16, 16, 16, 255);
    const ImU32 secondary_color = IM_COL32(120, 120, 120, 255);

    ImGui::TextColored(COL_ACCENT, "BER vs SNR");

    ImVec2 child_size(ImGui::GetContentRegionAvail().x, 300.0f);
    if (child_size.x < 420.0f) {
        child_size.x = 420.0f;
    }

    ImGui::BeginChild("ieee_ber_plot", child_size, true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 win_pos = ImGui::GetWindowPos();
    ImVec2 win_size = ImGui::GetWindowSize();

    ImVec2 canvas_min(win_pos.x + 10.0f, win_pos.y + 10.0f);
    ImVec2 canvas_max(win_pos.x + win_size.x - 10.0f, win_pos.y + win_size.y - 10.0f);
    draw_list->AddRectFilled(canvas_min, canvas_max, canvas_bg, 0.0f);
    draw_list->AddRect(canvas_min, canvas_max, axis_color, 0.0f, 0, 1.0f);

    float left_margin = 58.0f;
    float right_margin = 18.0f;
    float top_margin = 22.0f;
    float bottom_margin = 44.0f;

    float plot_left = canvas_min.x + left_margin;
    float plot_right = canvas_max.x - right_margin;
    float plot_top = canvas_min.y + top_margin;
    float plot_bottom = canvas_max.y - bottom_margin;

    double x_min = results.front().snr_db;
    double x_max = results.front().snr_db;
    double min_ber = 1.0;
    double max_ber = 1e-12;

    for (const SweepResult& result : results) {
        x_min = std::min(x_min, result.snr_db);
        x_max = std::max(x_max, result.snr_db);
        if (result.primary_ber > 0.0) {
            min_ber = std::min(min_ber, result.primary_ber);
            max_ber = std::max(max_ber, result.primary_ber);
        }
        if (has_secondary && result.secondary_ber > 0.0) {
            min_ber = std::min(min_ber, result.secondary_ber);
            max_ber = std::max(max_ber, result.secondary_ber);
        }
    }

    min_ber = std::max(min_ber, 1e-12);
    max_ber = std::max(max_ber, min_ber);
    double y_min = std::pow(10.0, std::floor(std::log10(min_ber)));
    double y_max = std::pow(10.0, std::ceil(std::log10(max_ber)));
    if (std::fabs(y_max - y_min) < 1e-18) {
        y_max *= 10.0;
        y_min /= 10.0;
    }

    int exp_lo = static_cast<int>(std::floor(std::log10(y_min)));
    int exp_hi = static_cast<int>(std::ceil(std::log10(y_max)));
    for (int exp = exp_lo; exp <= exp_hi; ++exp) {
        for (int digit = 1; digit <= 9; ++digit) {
            double yv = digit * std::pow(10.0, exp);
            if (yv < y_min * (1.0 - 1e-9) || yv > y_max * (1.0 + 1e-9)) {
                continue;
            }
            float y = map_plot_y(yv, y_min, y_max, plot_top, plot_bottom);
            bool major = (digit == 1);
            draw_list->AddLine(ImVec2(plot_left, y), ImVec2(plot_right, y),
                               major ? major_grid : minor_grid, major ? 1.0f : 0.8f);
            if (major) {
                std::ostringstream label;
                label << "1e" << exp;
                draw_list->AddText(ImVec2(canvas_min.x + 4.0f, y - 7.0f), text_color, label.str().c_str());
                draw_list->AddLine(ImVec2(plot_left, y), ImVec2(plot_left + 5.0f, y), axis_color, 1.0f);
            }
        }
    }

    int x_tick_step = std::max(1, static_cast<int>(results.size() / 6));
    for (size_t i = 0; i < results.size(); i += x_tick_step) {
        float x = map_plot_x(results[i].snr_db, x_min, x_max, plot_left, plot_right);
        draw_list->AddLine(ImVec2(x, plot_top), ImVec2(x, plot_bottom), minor_grid, 0.8f);
        draw_list->AddLine(ImVec2(x, plot_bottom), ImVec2(x, plot_bottom - 5.0f), axis_color, 1.0f);
        std::ostringstream label;
        label << std::fixed << std::setprecision(1) << results[i].snr_db;
        ImVec2 text_size = ImGui::CalcTextSize(label.str().c_str());
        draw_list->AddText(ImVec2(x - 0.5f * text_size.x, plot_bottom + 8.0f), text_color, label.str().c_str());
    }
    if ((results.size() - 1) % x_tick_step != 0) {
        const SweepResult& last = results.back();
        float x = map_plot_x(last.snr_db, x_min, x_max, plot_left, plot_right);
        draw_list->AddLine(ImVec2(x, plot_top), ImVec2(x, plot_bottom), minor_grid, 0.8f);
        draw_list->AddLine(ImVec2(x, plot_bottom), ImVec2(x, plot_bottom - 5.0f), axis_color, 1.0f);
        std::ostringstream label;
        label << std::fixed << std::setprecision(1) << last.snr_db;
        ImVec2 text_size = ImGui::CalcTextSize(label.str().c_str());
        draw_list->AddText(ImVec2(x - 0.5f * text_size.x, plot_bottom + 8.0f), text_color, label.str().c_str());
    }

    draw_list->AddRect(ImVec2(plot_left, plot_top), ImVec2(plot_right, plot_bottom), axis_color, 0.0f, 0, 1.0f);

    std::vector<ImVec2> primary_pts;
    std::vector<ImVec2> secondary_pts;
    primary_pts.reserve(results.size());
    secondary_pts.reserve(results.size());
    for (const SweepResult& result : results) {
        float x = map_plot_x(result.snr_db, x_min, x_max, plot_left, plot_right);
        primary_pts.emplace_back(x, map_plot_y(std::max(result.primary_ber, 1e-12), y_min, y_max, plot_top, plot_bottom));
        if (has_secondary) {
            secondary_pts.emplace_back(x, map_plot_y(std::max(result.secondary_ber, 1e-12), y_min, y_max, plot_top, plot_bottom));
        }
    }

    for (size_t i = 1; i < primary_pts.size(); ++i) {
        draw_list->AddLine(primary_pts[i - 1], primary_pts[i], primary_color, 1.3f);
    }
    for (const ImVec2& p : primary_pts) {
        draw_marker_circle(draw_list, p, primary_color);
    }

    if (has_secondary) {
        for (size_t i = 1; i < secondary_pts.size(); ++i) {
            draw_dashed_line(draw_list, secondary_pts[i - 1], secondary_pts[i], secondary_color, 1.3f);
        }
        for (const ImVec2& p : secondary_pts) {
            draw_marker_cross(draw_list, p, secondary_color);
        }
    }

    draw_list->AddText(ImVec2(plot_left + 4.0f, canvas_min.y + 2.0f), text_color, "BER");
    const char* xlabel = "SNR (dB)";
    ImVec2 xlabel_size = ImGui::CalcTextSize(xlabel);
    draw_list->AddText(ImVec2(0.5f * (plot_left + plot_right - xlabel_size.x), plot_bottom + 24.0f),
                       text_color, xlabel);

    draw_ieee_plot_legend(draw_list, plot_right - 150.0f, plot_top + 16.0f,
                          primary_label, secondary_label, has_secondary);

    ImGui::Dummy(child_size);
    ImGui::EndChild();
}


static void draw_results() {
    std::lock_guard<std::mutex> lock(pipeline_status.mutex);

    if (pipeline_status.results.empty()) {
        ImGui::TextColored(COL_TEXT_DIM, "No sweep results yet. Click 'Run SNR Sweep' to start.");
        return;
    }

    draw_ieee_ber_plot(pipeline_status.results,
                       pipeline_status.primary_curve_label,
                       pipeline_status.secondary_curve_label,
                       pipeline_status.has_secondary_curve);

    ImGui::Spacing();
    ImGui::TextColored(COL_ACCENT, "Results Table");

    if (ImGui::BeginTable("results_table",
                          pipeline_status.has_secondary_curve ? 7 : 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame,
                          ImVec2(0.0f, 220.0f))) {
        ImGui::TableSetupColumn("SNR (dB)");
        ImGui::TableSetupColumn(pipeline_status.primary_curve_label.c_str());
        if (pipeline_status.has_secondary_curve) {
            ImGui::TableSetupColumn(pipeline_status.secondary_curve_label.c_str());
        }
        ImGui::TableSetupColumn("Primary Errors");
        if (pipeline_status.has_secondary_curve) {
            ImGui::TableSetupColumn("Secondary Errors");
        }
        ImGui::TableSetupColumn("Packets");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        for (const SweepResult& result : pipeline_status.results) {
            std::ostringstream snr_str, primary_ber, secondary_ber;
            snr_str << std::fixed << std::setprecision(2) << result.snr_db;
            primary_ber << std::scientific << std::setprecision(2) << result.primary_ber;
            secondary_ber << std::scientific << std::setprecision(2) << result.secondary_ber;

            bool done = result.has_secondary
                ? (result.primary_hit_target && result.secondary_hit_target)
                : result.primary_hit_target;

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(snr_str.str().c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(primary_ber.str().c_str());
            int col = 2;
            if (pipeline_status.has_secondary_curve) {
                ImGui::TableSetColumnIndex(col++);
                ImGui::TextUnformatted(secondary_ber.str().c_str());
            }
            ImGui::TableSetColumnIndex(col++);
            ImGui::Text("%d", result.primary_bit_errors);
            if (pipeline_status.has_secondary_curve) {
                ImGui::TableSetColumnIndex(col++);
                ImGui::Text("%d", result.secondary_bit_errors);
            }
            ImGui::TableSetColumnIndex(col++);
            ImGui::Text("%d", result.total_blocks);
            ImGui::TableSetColumnIndex(col);
            if (done) {
                ImGui::TextColored(COL_ACCENT2, "%s", result.note.empty() ? "done" : result.note.c_str());
            } else if (!result.note.empty()) {
                ImGui::TextColored(COL_WARNING, "%s", result.note.c_str());
            } else {
                ImGui::TextColored(COL_TEXT_DIM, "running");
            }
        }

        ImGui::EndTable();
    }

    int total_blocks = 0;
    long long total_primary_errors = 0;
    long long total_secondary_errors = 0;
    for (const SweepResult& result : pipeline_status.results) {
        total_blocks += result.total_blocks;
        total_primary_errors += result.primary_bit_errors;
        total_secondary_errors += result.secondary_bit_errors;
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (pipeline_status.has_secondary_curve) {
        ImGui::TextColored(
            COL_TEXT_DIM,
            "Summary: %d SNR points, %d total packets, %s errors=%lld, %s errors=%lld",
            (int)pipeline_status.results.size(), total_blocks,
            pipeline_status.primary_curve_label.c_str(), total_primary_errors,
            pipeline_status.secondary_curve_label.c_str(), total_secondary_errors
        );
    } else {
        ImGui::TextColored(
            COL_TEXT_DIM,
            "Summary: %d SNR points, %d total packets, %s errors=%lld",
            (int)pipeline_status.results.size(), total_blocks,
            pipeline_status.primary_curve_label.c_str(), total_primary_errors
        );
    }
}

static void draw_dfec_dataset_tab() {
    std::string bundle_dir = normalize_bundle_path(config.dfec_bundle_dir);
    std::filesystem::path manifest_path = std::filesystem::path(bundle_dir) / "manifest.txt";
    bool has_manifest = !bundle_dir.empty() && file_exists(manifest_path.string().c_str());
    std::vector<double> pilots = selected_dfec_pilots(config);

    bool running = dfec_status.running;
    int exit_code = dfec_status.exit_code;
    std::string message;
    std::string output_csv;
    std::string raw_output;
    std::vector<DfecPacketResult> packets;
    std::vector<DfecPilotProgress> pilot_progress;
    {
        std::lock_guard<std::mutex> lock(dfec_status.mutex);
        message = dfec_status.message;
        output_csv = dfec_status.output_csv;
        raw_output = dfec_status.raw_output;
        packets = dfec_status.packets;
        pilot_progress = dfec_status.pilot_progress;
    }

    ImGui::TextColored(COL_ACCENT2, "C++ DFEC Dataset Decoder");
    ImGui::Separator();
    ImGui::TextColored(COL_TEXT_DIM,
                       "Drop a C++ DFEC bundle folder or its manifest.txt onto this window, or paste the path below.");
    ImGui::TextColored(COL_TEXT_DIM,
                       "Packet success means DFEC packet BER = 0 for that packet.");
    ImGui::Spacing();

    ImGui::InputText("Bundle Directory##dfec_bundle", config.dfec_bundle_dir, IM_ARRAYSIZE(config.dfec_bundle_dir));
    if (ImGui::Button("Use Default Bundle", ImVec2(180, 36))) {
        set_dfec_bundle_dir("DFEC_experiment/export/cpp_bundle");
        bundle_dir = normalize_bundle_path(config.dfec_bundle_dir);
        manifest_path = std::filesystem::path(bundle_dir) / "manifest.txt";
        has_manifest = !bundle_dir.empty() && file_exists(manifest_path.string().c_str());
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Path", ImVec2(140, 36))) {
        std::snprintf(config.dfec_bundle_dir, sizeof(config.dfec_bundle_dir), "%s", "");
        bundle_dir.clear();
        has_manifest = false;
    }

    if (has_manifest) {
        ImGui::TextColored(COL_ACCENT2, "Bundle ready: %s", bundle_dir.c_str());
    } else {
        ImGui::TextColored(COL_WARNING, "Bundle must contain manifest.txt plus the exported C++ binary matrices.");
    }

    ImGui::Spacing();
    ImGui::TextColored(COL_ACCENT2, "Pilot Ratio Choices");
    ImGui::Separator();
    for (int i = 0; i < DFEC_PILOT_OPTION_COUNT; ++i) {
        char label[32];
        std::snprintf(label, sizeof(label), "%.2f##pilot_%d", DFEC_PILOT_OPTIONS[i], i);
        ImGui::Checkbox(label, &config.dfec_pilot_selected[i]);
        if ((i % 3) != 2) {
            ImGui::SameLine();
        }
    }
    if (pilots.empty()) {
        ImGui::TextColored(COL_WARNING, "Select at least one pilot ratio.");
    } else {
        std::ostringstream pilot_ss;
        for (size_t i = 0; i < pilots.size(); ++i) {
            if (i != 0) {
                pilot_ss << ", ";
            }
            pilot_ss << std::fixed << std::setprecision(2) << pilots[i];
        }
        ImGui::TextColored(COL_TEXT_DIM, "Selected: %s", pilot_ss.str().c_str());
    }

    ImGui::Spacing();
    if (running) {
        ImGui::TextColored(COL_ACCENT2, "● Decode Status: Running...");
    } else if (exit_code == 0 && !packets.empty()) {
        ImGui::TextColored(COL_ACCENT2, "✓ Decode Status: Completed");
    } else if (exit_code > 0) {
        ImGui::TextColored(COL_WARNING, "✗ Decode Status: Failed (code %d)", exit_code);
    } else {
        ImGui::TextColored(COL_TEXT_DIM, "○ Decode Status: Ready");
    }
    if (!message.empty()) {
        ImGui::TextColored(COL_TEXT_DIM, "%s", message.c_str());
    }
    if (!output_csv.empty()) {
        ImGui::TextColored(COL_TEXT_DIM, "CSV: %s", output_csv.c_str());
    }

    bool decode_valid = has_manifest && !pilots.empty();
    if (!decode_valid || pipeline_status.running || running) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Run C++ DFEC Decode", ImVec2(260, 44))) {
        start_background_dfec_decode();
    }
    if (!decode_valid || pipeline_status.running || running) {
        ImGui::EndDisabled();
    }
    if (pipeline_status.running) {
        ImGui::TextColored(COL_TEXT_DIM, "SNR sweep is running. Dataset decode waits so we do not slow the sweep down.");
    }

    if (!pilot_progress.empty()) {
        int processed_total = 0;
        int success_total = 0;
        int expected_total = 0;
        for (const DfecPilotProgress& progress : pilot_progress) {
            processed_total += progress.processed_packets;
            success_total += progress.success_packets;
            expected_total += progress.total_packets;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(COL_ACCENT, "Pilot Summary");
        ImGui::TextColored(COL_TEXT_DIM, "Overall packet success rate: %.2f%% (%d / %d)",
                           processed_total > 0 ? 100.0f * static_cast<float>(success_total) / static_cast<float>(processed_total) : 0.0f,
                           success_total, processed_total);
        ImGui::TextColored(COL_TEXT_DIM, "Progress: %d / %d packet decodes",
                           processed_total, expected_total);

        if (ImGui::BeginTable("dfec_pilot_summary", 7,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp,
                              ImVec2(0.0f, 180.0f))) {
            ImGui::TableSetupColumn("Pilot");
            ImGui::TableSetupColumn("Pilot Bits");
            ImGui::TableSetupColumn("Processed");
            ImGui::TableSetupColumn("Success");
            ImGui::TableSetupColumn("Success Rate");
            ImGui::TableSetupColumn("Mean DFEC BER");
            ImGui::TableSetupColumn("Mean DFE BER");
            ImGui::TableHeadersRow();

            for (const DfecPilotProgress& progress : pilot_progress) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%.2f", progress.pilot_frac);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%d", progress.pilot_count);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%d / %d", progress.processed_packets, progress.total_packets);
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%d", progress.success_packets);
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%.2f%%", 100.0 * progress.success_rate);
                ImGui::TableSetColumnIndex(5);
                ImGui::Text("%.4e", progress.mean_dfec_ber);
                ImGui::TableSetColumnIndex(6);
                ImGui::Text("%.4e", progress.mean_dfe_ber);
            }
            ImGui::EndTable();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(COL_ACCENT, "Packet Results");
    if (packets.empty()) {
        ImGui::TextColored(COL_TEXT_DIM, "No packet results yet.");
    } else if (ImGui::BeginTable("dfec_packets", 8,
                                 ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                                 ImVec2(0.0f, 260.0f))) {
        ImGui::TableSetupColumn("Pilot");
        ImGui::TableSetupColumn("Packet");
        ImGui::TableSetupColumn("Frame");
        ImGui::TableSetupColumn("Block");
        ImGui::TableSetupColumn("DFEC BER");
        ImGui::TableSetupColumn("DFE BER");
        ImGui::TableSetupColumn("Success");
        ImGui::TableSetupColumn("Valid");
        ImGui::TableHeadersRow();

        const size_t start = packets.size() > 400 ? packets.size() - 400 : 0;
        for (size_t i = start; i < packets.size(); ++i) {
            const DfecPacketResult& packet = packets[i];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%.2f", packet.pilot_frac);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d / %d", packet.packet_index, packet.total_packets);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%d", packet.frame);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%d", packet.block);
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.4e", packet.dfec_ber);
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.4e", packet.dfe_ber);
            ImGui::TableSetColumnIndex(6);
            if (packet.success) {
                ImGui::TextColored(COL_ACCENT2, "yes");
            } else {
                ImGui::TextColored(COL_TEXT_DIM, "no");
            }
            ImGui::TableSetColumnIndex(7);
            ImGui::Text("%s", packet.joint_valid ? "yes" : "no");
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(COL_ACCENT, "Decoder Log");
    ImGui::BeginChild("dfec_log", ImVec2(0.0f, 220.0f), true);
    if (!raw_output.empty()) {
        ImGui::TextUnformatted(raw_output.c_str());
    } else {
        ImGui::TextColored(COL_TEXT_DIM, "No decoder log yet.");
    }
    ImGui::EndChild();
}

static void draw_main_window() {
    apply_pending_dropped_path();
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
    ImGui::Begin("LDPC BP Decoder GUI", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(COL_ACCENT, "LDPC Belief Propagation Decoder");
    ImGui::TextColored(COL_TEXT_DIM, "Code: n=2048, k=1024, rate=0.5, 3 checks/node, no 4-cycles");
    ImGui::Spacing();

    if (ImGui::BeginTabBar("MainTabs")) {
        if (ImGui::BeginTabItem("Configuration")) {
            ImGui::Spacing();
            ImGui::TextColored(COL_ACCENT2, "SNR Sweep Parameters");
            ImGui::Separator();
            ImGui::InputFloat("Start (dB)##snr_start", &config.snr_start_db);
            ImGui::InputFloat("Stop (dB)##snr_stop", &config.snr_stop_db);
            ImGui::InputFloat("Step (dB)##snr_step", &config.snr_step_db);
            ImGui::InputInt("Min Bit Errors Before Next SNR##target", &config.min_bit_errors);
            ImGui::InputInt("Max Packets Per SNR##max_packets", &config.max_packets);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::TextColored(COL_ACCENT2, "Channel Settings");
            ImGui::Separator();
            const char* channels[] = {"awgn", "isi"};
            if (ImGui::BeginCombo("Channel Type##channel_type", config.channel_type)) {
                for (int i = 0; i < IM_ARRAYSIZE(channels); ++i) {
                    bool is_selected = (std::strcmp(config.channel_type, channels[i]) == 0);
                    if (ImGui::Selectable(channels[i], is_selected)) {
                        std::strcpy(config.channel_type, channels[i]);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextColored(COL_TEXT_DIM, "SNR is automatically converted to AWGN noise std dev");
            if (std::strcmp(config.channel_type, "isi") == 0) {
                ImGui::Spacing();
                ImGui::TextColored(COL_TEXT_DIM,
                                   "ISI mode runs both joint dfec + prprp and dfe + prprp on the same packets");
                ImGui::TextColored(COL_TEXT_DIM,
                                   "Both BER curves are overlaid together in the results plot");
                ImGui::Spacing();
                ImGui::TextColored(COL_TEXT_DIM, "ISI Taps (Unit Power Normalized)");
                ImGui::SliderFloat("Tap 1##tap1", &config.isi_taps[0], 0.0f, 1.0f, "%.3f");
                if (ImGui::IsItemEdited()) normalize_isi_taps();
                ImGui::SliderFloat("Tap 2##tap2", &config.isi_taps[1], 0.0f, 1.0f, "%.3f");
                if (ImGui::IsItemEdited()) normalize_isi_taps();
                ImGui::SliderFloat("Tap 3##tap3", &config.isi_taps[2], 0.0f, 1.0f, "%.3f");
                if (ImGui::IsItemEdited()) normalize_isi_taps();

                float power = config.isi_taps[0] * config.isi_taps[0] +
                             config.isi_taps[1] * config.isi_taps[1] +
                             config.isi_taps[2] * config.isi_taps[2];
                ImGui::TextColored(COL_TEXT_DIM, "Channel Power: %.4f", power);
                ImGui::Spacing();
                draw_channel_plot();
            } else {
                ImGui::TextColored(COL_TEXT_DIM, "AWGN channel adds Gaussian noise only");
            }
            ImGui::Spacing();

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();

            bool config_valid = true;
            if (config.min_bit_errors <= 0) {
                ImGui::TextColored(COL_WARNING, "⚠ Minimum bit errors must be > 0");
                config_valid = false;
            }
            if (config.max_packets <= 0) {
                ImGui::TextColored(COL_WARNING, "⚠ Max packets must be > 0");
                config_valid = false;
            }
            if (config.snr_step_db <= 0.0f) {
                ImGui::TextColored(COL_WARNING, "⚠ SNR step must be > 0");
                config_valid = false;
            }
            if (config.snr_stop_db < config.snr_start_db) {
                ImGui::TextColored(COL_WARNING, "⚠ SNR stop must be ≥ SNR start");
                config_valid = false;
            }
            if (std::strcmp(config.channel_type, "isi") == 0
                && (config.isi_taps[0] < 0.0f || config.isi_taps[1] < 0.0f || config.isi_taps[2] < 0.0f)) {
                ImGui::TextColored(COL_WARNING, "⚠ ISI tap magnitudes must be non-negative");
                config_valid = false;
            }
            if (std::strcmp(config.channel_type, "isi") == 0
                && std::fabs(config.isi_taps[0]) < 1e-6f) {
                ImGui::TextColored(COL_WARNING, "⚠ Tap 1 must be non-zero for ISI decoding");
                config_valid = false;
            }

            ImGui::Spacing();
            ImGui::TextColored(COL_ACCENT2, "Simulation Control");
            ImGui::Separator();

            {
                bool is_running = pipeline_status.running;
                int exit_code = pipeline_status.exit_code;
                if (is_running) {
                    ImGui::TextColored(COL_ACCENT2, "● Status: Running...");
                } else if (exit_code == 0) {
                    ImGui::TextColored(COL_ACCENT2, "✓ Status: Completed");
                } else if (exit_code > 0) {
                    ImGui::TextColored(COL_WARNING, "✗ Status: Failed (code %d)", exit_code);
                } else {
                    ImGui::TextColored(COL_TEXT_DIM, "○ Status: Ready");
                }
            }

            ImGui::Spacing();

            bool dfec_running = dfec_status.running;

            if (!config_valid || dfec_running) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("▶ Run SNR Sweep", ImVec2(250, 45))) {
                {
                    std::lock_guard<std::mutex> lock(pipeline_status.mutex);
                    if (!pipeline_status.running) {
                        start_background_pipeline();
                    }
                }
            }

            if (!config_valid || dfec_running) {
                ImGui::EndDisabled();
            }

            if (dfec_running) {
                ImGui::TextColored(COL_TEXT_DIM, "C++ DFEC dataset decode is running. Sweep start is paused to keep performance unchanged.");
            }

            ImGui::SameLine();
            if (ImGui::Button("↻ Reset to Defaults", ImVec2(220, 45))) {
                config = UiConfig();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Results")) {
            ImGui::Spacing();
            draw_results();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("DFEC Dataset")) {
            ImGui::Spacing();
            draw_dfec_dataset_tab();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

int main(int, char**) {
    if (!glfwInit()) {
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1400, 800, "LDPC BP Decoder GUI", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetDropCallback(window, glfw_file_drop_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImFontConfig font_cfg;
    font_cfg.SizePixels = 18.0f;
    font_cfg.OversampleH = 2;
    io.Fonts->AddFontDefault(&font_cfg);

    apply_theme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_main_window();

        ImGui::Render();
        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(COL_BG.x, COL_BG.y, COL_BG.z, COL_BG.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (background_thread.joinable()) {
        background_thread.join();
    }
    if (dfec_thread.joinable()) {
        dfec_thread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
