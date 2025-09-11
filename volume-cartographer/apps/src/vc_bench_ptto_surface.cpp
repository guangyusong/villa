// Baseline benchmark for QuadSurface::pointTo
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <tuple>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/Surface.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

static bool is_tifxyz_dir(const fs::path& p) {
    return fs::exists(p/"x.tif") && fs::exists(p/"y.tif") && fs::exists(p/"z.tif");
}

static std::vector<fs::path> list_tifxyz_children(const fs::path& root) {
    std::vector<fs::path> out;
    if (!fs::exists(root) || !fs::is_directory(root)) return out;
    for (auto& de : fs::directory_iterator(root)) {
        if (de.is_directory() && is_tifxyz_dir(de.path())) out.push_back(de.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct Stats {
    size_t calls = 0;
    double time_ms = 0.0;
    double resid_sum = 0.0;
    double resid_max = 0.0;
    double err_sum = 0.0;
    double err_max = 0.0;
    size_t accepted = 0; // residual <= th
    int iterations = 0; // for sweep tracking
};

static inline bool is_invalid_vec3f(const cv::Vec3f& v) {
    return !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) ||
           (v[0] == -1.f && v[1] == -1.f && v[2] == -1.f);
}

int main(int argc, char** argv) {
    std::string root_dir;
    int samples_per_surface = 2000;
    int surfaces_limit = 0; // 0 = all
    unsigned int seed = 0;
    float threshold = 2.0f;
    int max_iters = 10;
    std::string save_csv_path;
    std::string use_samples_from;
    std::string save_samples_path;
    bool param_sweep = false;
    bool proposed_mode = false;       // Enable proposed speed/accuracy tweaks
    bool warmstart = true;            // Reuse pointer between nearby samples
    int pre_iters = 25;               // First-stage iterations in adaptive schedule
    bool compare = false;             // Run both baseline and proposed
    int coarse_stride = 0;            // 0=disabled; >0 enables coarse-grid pre-scan
    int sweep_min_iters = 10;
    int sweep_max_iters = 5000;
    std::string sweep_csv_path;
    // Proposed sweep lists (comma-separated). If set, evaluate all combinations for proposed mode
    std::string sweep_pre_iters_list_opt;
    std::string sweep_coarse_strides_list_opt;
    std::string sweep_warmstart_list_opt;

    po::options_description desc("Benchmark: QuadSurface::pointTo (baseline and proposed)");
    desc.add_options()
        ("help,h", "Print help")
        ("root,r", po::value<std::string>(&root_dir)->required(), "Root directory with tifxyz subdirs")
        ("samples,s", po::value<int>(&samples_per_surface)->default_value(samples_per_surface), "Samples per surface")
        ("surfaces", po::value<int>(&surfaces_limit)->default_value(surfaces_limit), "Max number of surfaces (0 = all)")
        ("seed", po::value<unsigned int>(&seed)->default_value(seed), "RNG seed (0 = nondeterministic)")
        ("threshold,t", po::value<float>(&threshold)->default_value(threshold), "Residual acceptance threshold")
        ("iters,i", po::value<int>(&max_iters)->default_value(max_iters), "Max iterations for pointTo")
        ("save-csv", po::value<std::string>(&save_csv_path), "Save per-surface metrics CSV")
        ("use-samples-from", po::value<std::string>(&use_samples_from), "Load fixed samples JSON from previous run")
        ("save-samples", po::value<std::string>(&save_samples_path), "Save chosen samples to JSON for reuse")
        ("param-sweep", po::bool_switch(&param_sweep), "Perform parameter sweep over iterations")
        ("sweep-min-iters", po::value<int>(&sweep_min_iters)->default_value(sweep_min_iters), "Min iterations for sweep")
        ("sweep-max-iters", po::value<int>(&sweep_max_iters)->default_value(sweep_max_iters), "Max iterations for sweep")
        ("sweep-csv", po::value<std::string>(&sweep_csv_path), "Save sweep results CSV")
        // Proposed-mode options
        ("proposed", po::bool_switch(&proposed_mode), "Enable proposed optimizations (warmstart + adaptive)")
        ("warmstart", po::value<bool>(&warmstart)->default_value(warmstart), "Reuse pointer between spatially-near samples")
        ("pre-iters", po::value<int>(&pre_iters)->default_value(pre_iters), "Adaptive schedule first-stage iterations")
        ("compare", po::bool_switch(&compare), "Run both baseline and proposed modes and print both summaries")
        ("coarse-stride", po::value<int>(&coarse_stride)->default_value(coarse_stride), "Coarse pre-scan stride (0=off)")
        // Proposed sweep options (comma-separated lists)
        ("sweep-pre-iters", po::value<std::string>(&sweep_pre_iters_list_opt), "Comma-separated pre-iters values for proposed sweep")
        ("sweep-coarse-strides", po::value<std::string>(&sweep_coarse_strides_list_opt), "Comma-separated coarse strides for proposed sweep")
        ("sweep-warmstart", po::value<std::string>(&sweep_warmstart_list_opt), "Comma-separated warmstart flags (0/1) for proposed sweep");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << std::endl; return 2;
    }

    auto dirs = list_tifxyz_children(root_dir);
    if (dirs.empty()) { std::cerr << "No tifxyz subdirectories under " << root_dir << std::endl; return 1; }
    if (surfaces_limit > 0 && surfaces_limit < (int)dirs.size()) dirs.resize(surfaces_limit);

    std::mt19937 rng(seed ? seed : (unsigned)std::random_device{}());
    if (seed) { std::srand(seed); } // Seed C RNG used internally by pointTo for reproducibility
    
    // Set up iteration values for sweep
    std::vector<int> iter_values;
    if (param_sweep) {
        // Create logarithmic sweep from min to max
        iter_values = {10, 25, 50, 100, 250, 500, 1000, 2500, 5000};
        // Filter to requested range
        iter_values.erase(
            std::remove_if(iter_values.begin(), iter_values.end(),
                [&](int v) { return v < sweep_min_iters || v > sweep_max_iters; }),
            iter_values.end());
        if (iter_values.empty()) iter_values = {sweep_min_iters, sweep_max_iters};
    } else {
        iter_values = {max_iters};
    }
    
    // Helpers to parse comma-separated lists
    auto parse_int_list = [](const std::string& s) {
        std::vector<int> out; if (s.empty()) return out; size_t i=0; while (i<s.size()) {
            while (i<s.size() && (s[i]==',' || isspace((unsigned char)s[i]))) ++i; if (i>=s.size()) break;
            size_t j=i; while (j<s.size() && s[j]!=',') ++j; try { out.push_back(std::stoi(s.substr(i,j-i))); } catch (...) {}
            i=j+1;
        } return out;
    };
    auto parse_bool_list = [&](const std::string& s){ auto v=parse_int_list(s); for(auto& x:v){ x = (x?1:0);} return v; };

    // Prepare proposed sweep combinations (fallback to single values if not provided)
    std::vector<int> sweep_pre_iters_list = parse_int_list(sweep_pre_iters_list_opt);
    std::vector<int> sweep_coarse_strides_list = parse_int_list(sweep_coarse_strides_list_opt);
    std::vector<int> sweep_warmstart_list = parse_bool_list(sweep_warmstart_list_opt);
    if (sweep_pre_iters_list.empty()) sweep_pre_iters_list = {pre_iters};
    if (sweep_coarse_strides_list.empty()) sweep_coarse_strides_list = {coarse_stride};
    if (sweep_warmstart_list.empty()) sweep_warmstart_list = {warmstart?1:0};
    const bool record_sweep = (sweep_pre_iters_list.size()>1 || sweep_coarse_strides_list.size()>1 || sweep_warmstart_list.size()>1);

    // Storage for results per-mode
    struct SweepRow { int iters; int pre_iters; int coarse_stride; int warmstart; Stats s; };
    std::vector<SweepRow> sweep_results_baseline;
    std::vector<SweepRow> sweep_results_proposed;
    
    Stats global_baseline, global_proposed;
    struct SurfRow { std::string name; std::string mode; Stats s; };
    std::vector<SurfRow> rows_baseline, rows_proposed;
    nlohmann::json samples_json = nlohmann::json::object();
    samples_json["surfaces"] = nlohmann::json::array();
    nlohmann::json loaded_samples;
    if (!use_samples_from.empty()) {
        try { std::ifstream in(use_samples_from); if (in) in >> loaded_samples; } catch (...) {}
    }

    for (const auto& d : dirs) {
        std::unique_ptr<QuadSurface> surf;
        try { surf.reset(load_quad_from_tifxyz(d.string())); }
        catch (...) { continue; }
        if (!surf) continue;
        cv::Mat_<cv::Vec3f>* pts = surf->rawPointsPtr();
        if (!pts || pts->empty()) continue;

        // Collect valid indices
        std::vector<cv::Point> valid;
        valid.reserve(pts->rows * pts->cols);
        for (int y = 1; y < pts->rows - 1; ++y) {
            for (int x = 1; x < pts->cols - 1; ++x) {
                if ((*pts)(y, x)[0] != -1.0f) valid.emplace_back(x, y);
            }
        }
        if (valid.empty()) continue;

        std::uniform_int_distribution<size_t> pick(0, valid.size() - 1);
        std::vector<cv::Point> samples_points;
        const std::string sname = d.filename().string();
        if (!use_samples_from.empty() && loaded_samples.contains("surfaces")) {
            for (const auto& sj : loaded_samples["surfaces"]) {
                if (sj.value("name", "") == sname && sj.contains("samples") && sj["samples"].is_array()) {
                    for (const auto& sp : sj["samples"]) if (sp.is_array() && sp.size()>=2) samples_points.emplace_back(sp[0].get<int>(), sp[1].get<int>());
                }
            }
        }
        if (samples_points.empty()) {
            samples_points.reserve(samples_per_surface);
            for (int k=0;k<samples_per_surface;++k) samples_points.push_back(valid[pick(rng)]);
        }
        if (!save_samples_path.empty()) {
            nlohmann::json s; s["name"] = sname; s["samples"] = nlohmann::json::array();
            for (auto& p : samples_points) s["samples"].push_back({p.x, p.y});
            samples_json["surfaces"].push_back(std::move(s));
        }
        // Warmup a few calls
        {
            cv::Vec3f ptr = surf->pointer();
            for (int w = 0; w < 8 && !valid.empty(); ++w) {
                auto p = valid[pick(rng)];
                (void)surf->pointTo(ptr, (*pts)(p.y,p.x), threshold, max_iters);
            }
        }

        // Prepare a spatially coherent ordering for proposed mode
        auto morton2D = [](uint32_t x, uint32_t y) {
            auto part1by1 = [](uint32_t n) {
                n &= 0x0000ffff;
                n = (n | (n << 8)) & 0x00FF00FF;
                n = (n | (n << 4)) & 0x0F0F0F0F;
                n = (n | (n << 2)) & 0x33333333;
                n = (n | (n << 1)) & 0x55555555;
                return n;
            };
            return (part1by1(y) << 1) | part1by1(x);
        };
        std::vector<cv::Point> samples_points_sorted = samples_points;
        if (proposed_mode || compare) {
            std::sort(samples_points_sorted.begin(), samples_points_sorted.end(), [&](const cv::Point& a, const cv::Point& b){
                return morton2D((uint32_t)a.x, (uint32_t)a.y) < morton2D((uint32_t)b.x, (uint32_t)b.y);
            });
        }

        // Helper to run either variant
        auto run_variant = [&](bool use_proposed,
                               Stats &global_out,
                               std::vector<SurfRow> &rows_out,
                               std::vector<SweepRow> &sweep_out,
                               int pre_iters_val,
                               int coarse_stride_val,
                               int warmstart_val){
            const std::vector<cv::Point>& seq = (use_proposed ? samples_points_sorted : samples_points);
            
            for (int iter_val : iter_values) {
                Stats local; local.iterations = iter_val;
                cv::Vec3f ptr_prev = surf->pointer();

                for (int k = 0; k < (int)seq.size(); ++k) {
                    auto p = seq[k];
                    cv::Vec3f tgt = (*pts)(p.y, p.x);

                    cv::Vec3f ptr = (use_proposed && warmstart_val) ? ptr_prev : surf->pointer();
                    // Start timing before any proposed prework
                    auto t0 = std::chrono::steady_clock::now();

                    // Optional coarse pre-scan to seed pointer closer to target (proposed mode only)
                    if (use_proposed && coarse_stride_val > 1) {
                        int best_x = -1, best_y = -1;
                        float best_d2 = std::numeric_limits<float>::infinity();
                        for (int yy = 1; yy < pts->rows - 1; yy += coarse_stride_val) {
                            for (int xx = 1; xx < pts->cols - 1; xx += coarse_stride_val) {
                                const cv::Vec3f& v = (*pts)(yy, xx);
                                if (v[0] == -1.f) continue;
                                float dx = v[0] - tgt[0];
                                float dy = v[1] - tgt[1];
                                float dz = v[2] - tgt[2];
                                float d2 = dx*dx + dy*dy + dz*dz;
                                if (d2 < best_d2) { best_d2 = d2; best_x = xx; best_y = yy; }
                            }
                        }
                        if (best_x >= 0) {
                            cv::Vec3f cur_int = surf->loc_raw(ptr);
                            cv::Vec2f sc = surf->scale();
                            cv::Vec3f delta_nominal = {(best_x - cur_int[0]) / sc[0], (best_y - cur_int[1]) / sc[1], 0};
                            surf->move(ptr, delta_nominal);
                        }
                    }
                    float residual;
                    if (use_proposed && pre_iters_val > 0 && iter_val > pre_iters_val) {
                        residual = surf->pointTo(ptr, tgt, threshold, pre_iters_val);
                        if (!(residual >= 0.0f && residual <= threshold)) {
                            residual = surf->pointTo(ptr, tgt, threshold, iter_val - pre_iters_val);
                        }
                    } else {
                        residual = surf->pointTo(ptr, tgt, threshold, iter_val);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    cv::Vec3f mapped = surf->coord(ptr);
                    if (is_invalid_vec3f(mapped)) continue;

                    double err = cv::norm(mapped - tgt);
                    local.calls++;
                    local.time_ms += dt_ms;
                    local.resid_sum += residual;
                    local.resid_max = std::max(local.resid_max, (double)residual);
                    local.err_sum += err;
                    local.err_max = std::max(local.err_max, err);
                    if (residual >= 0.0f && residual <= threshold) local.accepted++;

                    if (use_proposed && warmstart_val) ptr_prev = ptr;
                }

                // If this is the primary iteration value, save for per-surface stats
                if (iter_val == (param_sweep ? iter_values[0] : max_iters)) {
                    std::string mode_lbl = use_proposed ? (std::string("proposed(p=") + std::to_string(pre_iters_val) + ",s=" + std::to_string(coarse_stride_val) + ",w=" + std::to_string(warmstart_val) + ")") : "baseline";
                    rows_out.push_back({sname, mode_lbl, local});
                    global_out.calls += local.calls;
                    global_out.time_ms += local.time_ms;
                    global_out.resid_sum += local.resid_sum;
                    global_out.resid_max = std::max(global_out.resid_max, local.resid_max);
                    global_out.err_sum += local.err_sum;
                    global_out.err_max = std::max(global_out.err_max, local.err_max);
                    global_out.accepted += local.accepted;
                }

                // Store for sweep results
                if (record_sweep || param_sweep) {
                    auto it = std::find_if(sweep_out.begin(), sweep_out.end(),
                        [&](const SweepRow& r) { return r.iters == iter_val && r.pre_iters==pre_iters_val && r.coarse_stride==coarse_stride_val && r.warmstart==warmstart_val; });
                    if (it == sweep_out.end()) {
                        SweepRow row; row.iters=iter_val; row.pre_iters=pre_iters_val; row.coarse_stride=coarse_stride_val; row.warmstart=warmstart_val; row.s=local; sweep_out.push_back(row);
                    } else {
                        it->s.calls += local.calls;
                        it->s.time_ms += local.time_ms;
                        it->s.resid_sum += local.resid_sum;
                        it->s.resid_max = std::max(it->s.resid_max, local.resid_max);
                        it->s.err_sum += local.err_sum;
                        it->s.err_max = std::max(it->s.err_max, local.err_max);
                        it->s.accepted += local.accepted;
                    }
                }
            }
        };

        // Run the requested mode(s)
        if (compare) {
            run_variant(false, global_baseline, rows_baseline, sweep_results_baseline, pre_iters, coarse_stride, warmstart?1:0);
            for (int pi : sweep_pre_iters_list)
                for (int cs : sweep_coarse_strides_list)
                    for (int ws : sweep_warmstart_list)
                        run_variant(true,  global_proposed,  rows_proposed,  sweep_results_proposed, pi, cs, ws);
        } else {
            if (proposed_mode) {
                for (int pi : sweep_pre_iters_list)
                    for (int cs : sweep_coarse_strides_list)
                        for (int ws : sweep_warmstart_list)
                            run_variant(true,  global_proposed,  rows_proposed,  sweep_results_proposed, pi, cs, ws);
            } else {
                run_variant(false, global_baseline, rows_baseline, sweep_results_baseline, pre_iters, coarse_stride, warmstart?1:0);
            }
        }
    }

    const bool ran_baseline = compare || !proposed_mode;
    const bool ran_proposed = compare || proposed_mode;
    if ((!ran_baseline || global_baseline.calls == 0) && (!ran_proposed || global_proposed.calls == 0)) {
        std::cerr << "No valid samples processed." << std::endl; return 1;
    }

    auto print_summary = [&](const char* title, const Stats& s){
        if (s.calls == 0) return;
        std::cout << title << "\n";
        std::cout << "calls: " << s.calls << "\n";
        std::cout << "time_avg_ms: " << (s.time_ms / s.calls) << "\n";
        std::cout << "resid_avg: " << (s.resid_sum / s.calls) << ", resid_max: " << s.resid_max << "\n";
        std::cout << "error_avg: " << (s.err_sum / s.calls) << ", error_max: " << s.err_max << "\n";
        std::cout << "accept_percent: " << (100.0 * (double)s.accepted / (double)s.calls) << "\n";
    };

    if (ran_baseline) print_summary("QuadSurface::pointTo baseline", global_baseline);
    if (ran_proposed) print_summary("\nQuadSurface::pointTo proposed", global_proposed);

    if (ran_baseline && ran_proposed && global_baseline.calls>0 && global_proposed.calls>0) {
        auto avg = [](double sum, size_t n){ return n? sum / n : 0.0; };
        double b_time = avg(global_baseline.time_ms, global_baseline.calls);
        double p_time = avg(global_proposed.time_ms, global_proposed.calls);
        double b_resid = avg(global_baseline.resid_sum, global_baseline.calls);
        double p_resid = avg(global_proposed.resid_sum, global_proposed.calls);
        double b_err = avg(global_baseline.err_sum, global_baseline.calls);
        double p_err = avg(global_proposed.err_sum, global_proposed.calls);
        double b_acc = 100.0 * (double)global_baseline.accepted / (double)global_baseline.calls;
        double p_acc = 100.0 * (double)global_proposed.accepted / (double)global_proposed.calls;

        std::cout << "\nSide-by-side (baseline vs proposed)\n";
        std::cout << "time_avg_ms:    " << b_time << "\t" << p_time << "\t(speedup x" << (b_time>0? b_time/p_time : 0) << ")\n";
        std::cout << "resid_avg:      " << b_resid << "\t" << p_resid << "\n";
        std::cout << "error_avg:      " << b_err << "\t" << p_err << "\n";
        std::cout << "accept_percent: " << b_acc << "\t" << p_acc << "\n";
    }

    // Print a compact leaderboard for proposed parameter combos if we collected sweep rows
    if (ran_proposed && !sweep_results_proposed.empty()) {
        struct Agg {
            size_t calls=0; double time_ms=0, resid_sum=0, resid_max=0, err_sum=0, err_max=0; size_t accepted=0;
        };
        struct Key { int p; int s; int w; bool operator<(const Key& o) const { return std::tie(p,s,w) < std::tie(o.p,o.s,o.w); } };
        std::map<Key, Agg> agg;
        for (const auto& sr : sweep_results_proposed) {
            Key k{sr.pre_iters, sr.coarse_stride, sr.warmstart}; Agg &a = agg[k];
            a.calls += sr.s.calls; a.time_ms += sr.s.time_ms; a.resid_sum += sr.s.resid_sum; a.err_sum += sr.s.err_sum; a.accepted += sr.s.accepted;
            a.resid_max = std::max(a.resid_max, sr.s.resid_max); a.err_max = std::max(a.err_max, sr.s.err_max);
        }
        struct Row { int p,s,w; double tavg,ravg,eavg,apct; double rmax,emax; size_t calls; };
        std::vector<Row> rows;
        rows.reserve(agg.size());
        for (auto& it : agg) {
            const Key& k = it.first; const Agg& a = it.second; if (!a.calls) continue;
            rows.push_back({k.p,k.s,k.w, a.time_ms/a.calls, a.resid_sum/a.calls, a.err_sum/a.calls, (100.0*(double)a.accepted/(double)a.calls), a.resid_max, a.err_max, a.calls});
        }
        auto print_top = [&](const char* title, auto comp){
            std::sort(rows.begin(), rows.end(), comp);
            std::cout << "\nBest combos — " << title << "\n";
            std::cout << "pre-iters\tcoarse-stride\twarmstart\tcalls\ttime_avg_ms\tresid_avg\terror_avg\taccept%\n";
            int printed = 0; for (const auto& r : rows) { if (printed++>=12) break; std::cout << r.p << "\t" << r.s << "\t" << r.w << "\t" << r.calls << "\t" << r.tavg << "\t" << r.ravg << "\t" << r.eavg << "\t" << r.apct << "\n"; }
        };
        // By speed (ascending time)
        print_top("lowest time", [](const Row& a, const Row& b){ return a.tavg < b.tavg; });
        // By error (ascending)
        print_top("lowest error", [](const Row& a, const Row& b){ return a.eavg < b.eavg; });
        // By acceptance (descending)
        print_top("highest accept%", [](const Row& a, const Row& b){ return a.apct > b.apct; });
    }

    if (!save_csv_path.empty()) {
        try {
            std::ofstream c(save_csv_path);
            if (c) {
                c << "surface,mode,calls,time_avg_ms,resid_avg,resid_max,error_avg,error_max,accept_percent\n";
                auto dump_rows = [&](const std::vector<SurfRow>& rows){
                    for (const auto& r : rows) {
                        const Stats& s = r.s;
                        double tavg = (s.calls? s.time_ms / s.calls : 0.0);
                        double ravg = (s.calls? s.resid_sum / s.calls : 0.0);
                        double eavg = (s.calls? s.err_sum / s.calls : 0.0);
                        double apct = (s.calls? (100.0 * (double)s.accepted / (double)s.calls) : 0.0);
                        c << r.name << "," << r.mode << "," << s.calls << "," << tavg << "," << ravg << "," << s.resid_max
                          << "," << eavg << "," << s.err_max << "," << apct << "\n";
                    }
                };
                if (ran_baseline) dump_rows(rows_baseline);
                if (ran_proposed) dump_rows(rows_proposed);
                // Summary rows per mode
                auto dump_summary = [&](const char* mode, const Stats& s){
                    if (s.calls == 0) return;
                    double tavg = s.time_ms / s.calls;
                    double ravg = s.resid_sum / s.calls;
                    double eavg = s.err_sum / s.calls;
                    double apct = 100.0 * (double)s.accepted / (double)s.calls;
                    c << "SUMMARY," << mode << "," << s.calls << "," << tavg << "," << ravg << "," << s.resid_max
                      << "," << eavg << "," << s.err_max << "," << apct << "\n";
                };
                if (ran_baseline) dump_summary("baseline", global_baseline);
                if (ran_proposed) dump_summary("proposed", global_proposed);
            }
        } catch (...) { std::cerr << "Failed to save CSV: " << save_csv_path << std::endl; }
    }

    if (!save_samples_path.empty()) {
        try { std::ofstream o(save_samples_path); if (o) o << samples_json.dump(2); } catch (...) { std::cerr << "Failed to save samples JSON: " << save_samples_path << std::endl; }
    }
    
    // Output sweep results if requested
    if (param_sweep) {
        auto print_sweep = [&](const char* mode, const std::vector<SweepRow>& v){
            if (v.empty()) return;
            std::cout << "\nParameter Sweep Results (" << mode << "):\n";
            std::cout << "iters\tpre-iters\tcoarse-stride\twarmstart\ttime_avg_ms\tresid_avg\terror_avg\taccept_percent\n";
            for (const auto& sr : v) {
                const Stats& s = sr.s;
                if (s.calls > 0) {
                    double tavg = s.time_ms / s.calls;
                    double ravg = s.resid_sum / s.calls;
                    double eavg = s.err_sum / s.calls;
                    double apct = 100.0 * (double)s.accepted / (double)s.calls;
                    std::cout << sr.iters << "\t" << sr.pre_iters << "\t" << sr.coarse_stride << "\t" << sr.warmstart
                              << "\t" << tavg << "\t" << ravg << "\t" 
                              << eavg << "\t" << apct << "\n";
                }
            }
        };
        if (ran_baseline) print_sweep("baseline", sweep_results_baseline);
        if (ran_proposed) print_sweep("proposed", sweep_results_proposed);

        if (!sweep_csv_path.empty()) {
            try {
                std::ofstream sc(sweep_csv_path);
                if (sc) {
                    sc << "mode,iterations,pre_iters,coarse_stride,warmstart,calls,time_avg_ms,resid_avg,resid_max,error_avg,error_max,accept_percent\n";
                    auto dump_sweep = [&](const char* mode, const std::vector<SweepRow>& v){
                        for (const auto& sr : v) {
                            const Stats& s = sr.s;
                            if (s.calls > 0) {
                                double tavg = s.time_ms / s.calls;
                                double ravg = s.resid_sum / s.calls;
                                double eavg = s.err_sum / s.calls;
                                double apct = 100.0 * (double)s.accepted / (double)s.calls;
                                sc << mode << "," << sr.iters << "," << sr.pre_iters << "," << sr.coarse_stride << "," << sr.warmstart
                                   << "," << s.calls << "," << tavg << "," << ravg
                                   << "," << s.resid_max << "," << eavg << "," << s.err_max << "," << apct << "\n";
                            }
                        }
                    };
                    if (ran_baseline) dump_sweep("baseline", sweep_results_baseline);
                    if (ran_proposed) dump_sweep("proposed", sweep_results_proposed);
                }
            } catch (...) { std::cerr << "Failed to save sweep CSV: " << sweep_csv_path << std::endl; }
        }
    }
    
    return 0;
}
