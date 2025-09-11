// vc_benchmark_pointto.cpp
// Benchmark QuadSurface::pointTo speed and accuracy on a set of tifxyz surfaces.

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <boost/program_options.hpp>

#include "vc/core/util/Surface.hpp"
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace fs = std::filesystem;
namespace po = boost::program_options;

static bool is_tifxyz_dir(const fs::path& p) {
    return fs::exists(p/"x.tif") && fs::exists(p/"y.tif") && fs::exists(p/"z.tif");
}

struct Stats {
    size_t calls = 0;
    double time_ms = 0.0;       // accumulated ms spent inside pointTo
    double probe_ms = 0.0;      // accumulated ms spent inside coarse probe (if enabled)
    double residual_sum = 0.0;  // sum of returned distances
    double err_sum = 0.0;       // sum of coord error to target
    double residual_max = 0.0;
    double err_max = 0.0;
    double recomputed_sum = 0.0;   // sum of residual recomputed from raw points at solved loc
    double recomputed_max = 0.0;
    double discrep_sum = 0.0;      // sum of |returned_residual - recomputed|
    double discrep_max = 0.0;
    size_t accepted_calls = 0;     // count of accepted@threshold
};

static void update_stats(Stats& s, double dt_ms, double residual, double err) {
    s.calls++;
    s.time_ms += dt_ms;
    s.residual_sum += residual;
    s.err_sum += err;
    s.residual_max = std::max(s.residual_max, residual);
    s.err_max = std::max(s.err_max, err);
}

static void update_recomputed(Stats& s, double recomputed, double discrep) {
    s.recomputed_sum += recomputed;
    s.recomputed_max = std::max(s.recomputed_max, recomputed);
    s.discrep_sum += discrep;
    s.discrep_max = std::max(s.discrep_max, discrep);
}

static inline void update_probe_time(Stats& s, double dt_ms) {
    s.probe_ms += dt_ms;
}

static std::vector<fs::path> list_tifxyz_children(const fs::path& root, bool recursive) {
    std::vector<fs::path> out;
    if (!fs::exists(root) || !fs::is_directory(root)) return out;
    if (!recursive) {
        for (auto& de : fs::directory_iterator(root)) {
            if (de.is_directory() && is_tifxyz_dir(de.path())) out.push_back(de.path());
        }
    } else {
        for (auto& de : fs::recursive_directory_iterator(root)) {
            if (de.is_directory() && is_tifxyz_dir(de.path())) out.push_back(de.path());
        }
    }
    return out;
}

int main(int argc, char** argv) {
    // Options
    std::string root_dir;
    int samples_per_surface = 2000;
    float noise_std = 0.0f;
    float threshold = 2.0f;
    int max_iters = 100;
    bool recursive = false;
    unsigned int seed = 0;
    std::string init_mode = "center"; // center|random|truth
    std::string mode = "self";        // self|multisurface
    bool percentiles = false;
    bool coarse_probe = false;  // perform a coarse probe to seed ptr
    bool adaptive_probe = false; // quick pre-probe refine to decide if coarse probe is needed
    int probe_stride = 64;      // grid stride in pixels (ignored if probe_samples>0)
    int probe_samples = 0;      // random candidate samples; if >0, used instead of grid
    int preprobe_iters = 3;
    float preprobe_th = 4.0f;
    std::string save_json_path;
    std::string use_samples_from;
    std::string compare_to_path;
    int num_surfaces = 0;       // if >0, pick this many tifxyz dirs deterministically

    po::options_description desc("Benchmark QuadSurface::pointTo over tifxyz directories");
    desc.add_options()
        ("help,h", "Print help")
        ("root,r", po::value<std::string>(&root_dir)->required(), "Root directory containing tifxyz subdirectories")
        ("samples,s", po::value<int>(&samples_per_surface)->default_value(samples_per_surface), "Samples per surface")
        ("noise,n", po::value<float>(&noise_std)->default_value(noise_std), "Gaussian noise stddev added to targets (in world units)")
        ("threshold,t", po::value<float>(&threshold)->default_value(threshold), "Convergence threshold for pointTo")
        ("iters,i", po::value<int>(&max_iters)->default_value(max_iters), "Max iterations for pointTo")
        ("recursive", po::bool_switch(&recursive), "Recurse into subdirectories to find tifxyz")
        ("seed", po::value<unsigned int>(&seed)->default_value(seed), "RNG seed (0 = nondeterministic)")
        ("init", po::value<std::string>(&init_mode)->default_value(init_mode), "Init mode: center|random|truth")
        ("mode", po::value<std::string>(&mode)->default_value(mode), "Benchmark mode: self|multisurface")
        ("percentiles", po::bool_switch(&percentiles), "Print p50/p95 for time/residual/error")
        ("test", po::value<std::string>()->default_value(""), "Preset: growsurf|cwindow|both")
        ("coarse-probe", po::bool_switch(&coarse_probe), "Enable coarse probe seeding before pointTo")
        ("probe-stride", po::value<int>(&probe_stride)->default_value(probe_stride), "Coarse probe grid stride (pixels)")
        ("probe-samples", po::value<int>(&probe_samples)->default_value(probe_samples), "Coarse probe random samples (overrides stride) ")
        ("adaptive-probe", po::bool_switch(&adaptive_probe), "Enable adaptive pre-probe refine before coarse probe")
        ("preprobe-iters", po::value<int>(&preprobe_iters)->default_value(preprobe_iters), "Iterations for adaptive pre-probe refine")
        ("preprobe-th", po::value<float>(&preprobe_th)->default_value(preprobe_th), "Threshold for adaptive pre-probe refine")
        ("save-json", po::value<std::string>(&save_json_path), "Save per-surface metrics and samples to JSON file")
        ("use-samples-from", po::value<std::string>(&use_samples_from), "Use samples from a previous JSON (ensures A/B consistency)")
        ("compare-to", po::value<std::string>(&compare_to_path), "Compare current run against metrics JSON and print deltas")
        ("surfaces", po::value<int>(&num_surfaces)->default_value(num_surfaces), "Select this many tifxyz subdirs deterministically using --seed");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << std::endl; return 0; }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << std::endl; return 2;
    }

    auto tifxyz_dirs = list_tifxyz_children(root_dir, recursive);
    if (tifxyz_dirs.empty()) {
        std::cerr << "No tifxyz directories found under: " << root_dir << std::endl; return 1;
    }

    // Deterministic selection of a subset of surfaces if requested
    std::sort(tifxyz_dirs.begin(), tifxyz_dirs.end());
    if (num_surfaces > 0 && num_surfaces < (int)tifxyz_dirs.size()) {
        std::mt19937 pick_rng(seed ? seed : (unsigned)std::random_device{}());
        std::shuffle(tifxyz_dirs.begin(), tifxyz_dirs.end(), pick_rng);
        tifxyz_dirs.resize(num_surfaces);
        std::sort(tifxyz_dirs.begin(), tifxyz_dirs.end());
    }

    std::mt19937 rng(seed ? seed : (unsigned)std::random_device{}());
    std::normal_distribution<float> gauss(0.0f, std::max(0.0f, noise_std));

    // Helper percentiles over a vector of numbers
    
    auto push_vals = [&](const std::vector<double>& src, double& p50, double& p95) {
        if (src.empty()) { p50 = p95 = 0.0; return; }
        std::vector<double> tmp = src;
        std::sort(tmp.begin(), tmp.end());
        auto idx50 = static_cast<size_t>(0.5 * (tmp.size()-1));
        auto idx95 = static_cast<size_t>(0.95 * (tmp.size()-1));
        p50 = tmp[idx50];
        p95 = tmp[idx95];
    };

    auto is_invalid_vec3f = [](const cv::Vec3f& v) {
        return !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) ||
               (v[0] == -1.f && v[1] == -1.f && v[2] == -1.f);
    };

    // Bilinear sampling on raw points
    auto bilinear_at = [](const cv::Mat_<cv::Vec3f>& m, float x, float y, bool& ok) -> cv::Vec3f {
        ok = false;
        if (m.empty()) return {NAN,NAN,NAN};
        if (x < 0.0f || y < 0.0f || x > (float)(m.cols - 2) || y > (float)(m.rows - 2)) return {NAN,NAN,NAN};
        int x0 = (int)std::floor(x);
        int y0 = (int)std::floor(y);
        float dx = x - x0;
        float dy = y - y0;
        auto p00 = m(y0,   x0);
        auto p10 = m(y0,   x0+1);
        auto p01 = m(y0+1, x0);
        auto p11 = m(y0+1, x0+1);
        auto bad = [](const cv::Vec3f& v){return !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) || (v[0]==-1.f && v[1]==-1.f && v[2]==-1.f);};
        if (bad(p00) || bad(p10) || bad(p01) || bad(p11)) return {NAN,NAN,NAN};
        cv::Vec3f a = p00*(1.0f-dx) + p10*dx;
        cv::Vec3f b = p01*(1.0f-dx) + p11*dx;
        cv::Vec3f v = a*(1.0f-dy) + b*dy;
        ok = true;
        return v;
    };

    struct TestProfile { std::string name; std::string init; float th; int iters; };
    std::vector<TestProfile> tests;
    bool sweep_mode = false;
    {
        std::string preset = vm["test"].as<std::string>();
        if (preset == "benchmark") {
            sweep_mode = true;
            // We'll run a sweep later and return.
            // For sweep, we still want baseline defaults for reference.
            tests.push_back({"baseline", "center", 2.0f, 10});
            coarse_probe = false;
            adaptive_probe = false;
        } else if (preset == "baseline") {
            // Baseline: match GrowSurface defaults exactly, and ensure probes are off
            tests.push_back({"baseline", "center", 2.0f, 10});
            coarse_probe = false;
            adaptive_probe = false;
        } else if (preset == "growsurf") {
            tests.push_back({"growsurf", "center", 2.0f, 10});
        } else if (preset == "cwindow") {
            tests.push_back({"cwindow", "center", 4.0f, 100});
        } else if (preset == "both") {
            tests.push_back({"growsurf", "center", 2.0f, 10});
            tests.push_back({"cwindow", "center", 4.0f, 100});
        } else {
            tests.push_back({"custom", init_mode, threshold, max_iters});
        }
    }

    // Auto-name JSON output if not provided (after selecting surfaces and profiles)
    if (save_json_path.empty()) {
        const std::string root_tag = fs::path(root_dir).filename().string();
        const int sel_count = (int)tifxyz_dirs.size();
        std::string prof_tag;
        if (sweep_mode) prof_tag = "sweep"; else if (tests.size() == 1) prof_tag = tests[0].name; else prof_tag = "both";
        std::string probe_tag;
        if (adaptive_probe) probe_tag += "_apre" + std::to_string(preprobe_iters) + "x" + std::to_string((int)preprobe_th);
        if (coarse_probe) {
            if (probe_samples > 0) probe_tag += "_probeS" + std::to_string(probe_samples);
            else probe_tag += "_probeG" + std::to_string(std::max(1, probe_stride));
        }
        std::string params_tag;
        if (sweep_mode) {
            params_tag = "_profiles-sweep";
        } else if (tests.size() == 1) {
            params_tag = "_init-" + tests[0].init + "_th-" + std::to_string((int)tests[0].th) + "_it-" + std::to_string(tests[0].iters);
        } else {
            params_tag = "_profiles-both";
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf), "ptto_%s_%s%s%s_seed%u_surfs%d_samp%d.json",
                      prof_tag.c_str(), root_tag.c_str(), params_tag.c_str(), probe_tag.c_str(),
                      seed, sel_count, samples_per_surface);
        save_json_path = buf;
        std::cout << "Auto save JSON as: " << save_json_path << std::endl;
    }

    // Load samples from previous run if requested
    nlohmann::json loaded_samples;
    if (!use_samples_from.empty()) {
        try {
            std::ifstream in(use_samples_from);
            if (in) in >> loaded_samples;
        } catch (...) {}
    }

    // Accumulate JSON results
    nlohmann::json run_json = nlohmann::json::object();
    run_json["root"] = root_dir;
    run_json["coarse_probe"] = coarse_probe;
    run_json["probe_stride"] = probe_stride;
    run_json["probe_samples"] = probe_samples;
    run_json["seed"] = seed;
    run_json["profiles"] = nlohmann::json::array();

    // Special benchmark sweep: run multiple configs and report best differences
    if (sweep_mode) {
        struct RunCfg { std::string name; TestProfile prof; bool useAdaptive; int preIters; float preTh; bool useCoarse; int stride; int samples; bool useGrid; };
        std::vector<RunCfg> cfgs = {
            {"baseline",      {"baseline","center",2.0f,10}, false, 0, 0.0f, false, 0,   0,   false},
            {"probeG64",      {"growsurf","center",2.0f,10}, false, 0, 0.0f, true,  64,  0,   false},
            {"probeS500",     {"growsurf","center",2.0f,10}, false, 0, 0.0f, true,  0,   500, false},
            {"adaptive",      {"growsurf","center",2.0f,10}, true,  3, 4.0f, false, 0,   0,   false},
            {"adaptive+S500", {"growsurf","center",2.0f,10}, true,  3, 4.0f, true,  0,   500, false},
            // Grid-based free-function variants (no adaptive/coarse probe)
            {"grid",          {"grid","center",2.0f,10},      false, 0, 0.0f, false, 0,   0,   true},
            {"gridI1000",     {"grid","center",2.0f,1000},    false, 0, 0.0f, false, 0,   0,   true}
        };

        struct BenchRes {
            std::string name;
            double combined_avg_ms;
            double time_avg_ms;
            double accept_percent;
            double resid_p95;
            double resid_avg;
            double resid_max;
            double error_avg;
            double error_max;
        };
        std::vector<BenchRes> results;

        // Fixed samples per surface for consistency
        std::unordered_map<std::string, std::vector<cv::Point>> fixed_samples;
        // If using baseline JSON, reuse its samples per surface to ensure strict A/B
        if (!use_samples_from.empty() && loaded_samples.contains("profiles") && loaded_samples["profiles"].is_array()) {
            const nlohmann::json* prof_src = nullptr;
            for (const auto& pj : loaded_samples["profiles"]) { if (pj.value("name","baseline")=="baseline") { prof_src=&pj; break; } }
            if (!prof_src) prof_src = &loaded_samples["profiles"][0];
            if (prof_src && prof_src->contains("surfaces")) {
                for (const auto& s : (*prof_src)["surfaces"]) {
                    const std::string nm = s.value("name","");
                    if (s.contains("samples") && s["samples"].is_array()) {
                        std::vector<cv::Point> pts_vec; pts_vec.reserve(s["samples"].size());
                        for (const auto& sp : s["samples"]) if (sp.is_array() && sp.size()>=2) pts_vec.emplace_back(sp[0].get<int>(), sp[1].get<int>());
                        fixed_samples[nm] = std::move(pts_vec);
                    }
                }
            }
        }

        auto run_one = [&](const RunCfg& Cfg) {
            // Global accumulators
            Stats g;
            std::vector<double> grs; grs.reserve(tifxyz_dirs.size()*samples_per_surface);
            std::mt19937 rng_local(seed ? seed : (unsigned)std::random_device{}());

            for (const auto& dir : tifxyz_dirs) {
                std::unique_ptr<QuadSurface> surf;
                try { surf.reset(load_quad_from_tifxyz(dir.string())); } catch (...) { continue; }
                if (!surf) continue;
                cv::Mat_<cv::Vec3f>* pts = surf->rawPointsPtr();
                if (!pts || pts->empty()) continue;

                std::vector<cv::Point> valid;
                for (int y = 1; y < pts->rows - 1; ++y) for (int x = 1; x < pts->cols - 1; ++x) if ((*pts)(y,x)[0] != -1.0f) valid.emplace_back(x,y);
                if (valid.empty()) continue;

                // Samples for this surface
                std::string sname = dir.filename().string();
                auto itS = fixed_samples.find(sname);
                std::vector<cv::Point> samples_points;
                if (itS != fixed_samples.end()) samples_points = itS->second; else {
                    std::uniform_int_distribution<size_t> pick(0, valid.size()-1);
                    samples_points.reserve(samples_per_surface);
                    for (int k=0;k<samples_per_surface;++k) samples_points.push_back(valid[pick(rng_local)]);
                    fixed_samples.emplace(sname, samples_points);
                }

                std::uniform_int_distribution<int> xrand(1, pts->cols - 3), yrand(1, pts->rows - 3);

                Stats local;
                for (const auto& p : samples_points) {
                    cv::Vec3f tgt = (*pts)(p.y, p.x);

                    // Grid-based variant: call free function directly and bilinear-map result
                    if (Cfg.useGrid) {
                        auto t0 = std::chrono::steady_clock::now();
                        cv::Vec2f loc;
                        float residual = pointTo(loc, *pts, tgt, Cfg.prof.th, Cfg.prof.iters, surf->scale()[0]);
                        auto t1 = std::chrono::steady_clock::now();
                        double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                        bool ok = false; cv::Vec3f mapped = bilinear_at(*pts, loc[0], loc[1], ok);
                        if (ok) {
                            double err = cv::norm(mapped - tgt);
                            update_stats(local, dt_ms, residual, err);
                            if (residual >= 0.0f && residual <= Cfg.prof.th) local.accepted_calls++;
                            grs.push_back(residual);
                        }
                        continue;
                    }

                    cv::Vec3f ptr;
                    ptr = surf->pointer();

                    // adaptive preprobe
                    bool skip_coarse = false;
                    if (Cfg.useAdaptive) {
                        auto t0 = std::chrono::steady_clock::now();
                        cv::Vec3f tmp = ptr;
                        float r0 = surf->pointTo(tmp, tgt, Cfg.preTh, Cfg.preIters);
                        auto t1 = std::chrono::steady_clock::now();
                        update_probe_time(local, std::chrono::duration<double, std::milli>(t1 - t0).count());
                        if (r0 >= 0.0f && r0 <= Cfg.preTh) { ptr = tmp; skip_coarse = true; }
                    }

                    if (Cfg.useCoarse && !skip_coarse) {
                        auto tpb0 = std::chrono::steady_clock::now();
                        double best_rr = 1e30; float best_x = ptr[0] + pts->cols/2.0f, best_y = ptr[1] + pts->rows/2.0f;
                        if (Cfg.samples > 0) {
                            std::uniform_int_distribution<size_t> pick(0, valid.size()-1);
                            for (int s=0;s<Cfg.samples;++s) { auto qp=valid[pick(rng_local)]; bool ok; auto v=bilinear_at(*pts,(float)qp.x,(float)qp.y,ok); if(!ok) continue; double rr=cv::norm(v-tgt); if(rr<best_rr){best_rr=rr;best_x=(float)qp.x;best_y=(float)qp.y;} }
                        } else {
                            int stride = Cfg.stride>0?Cfg.stride:64;
                            for (int y=1;y<pts->rows-2;y+=stride) for (int x=1;x<pts->cols-2;x+=stride){ bool ok; auto v=bilinear_at(*pts,(float)x,(float)y,ok); if(!ok) continue; double rr=cv::norm(v-tgt); if(rr<best_rr){best_rr=rr;best_x=(float)x;best_y=(float)y;} }
                        }
                        ptr[0] = best_x - pts->cols/2.0f; ptr[1] = best_y - pts->rows/2.0f;
                        auto tpb1 = std::chrono::steady_clock::now();
                        update_probe_time(local, std::chrono::duration<double, std::milli>(tpb1 - tpb0).count());
                    }

                    auto t0 = std::chrono::steady_clock::now();
                    float residual = surf->pointTo(ptr, tgt, Cfg.prof.th, Cfg.prof.iters);
                    auto t1 = std::chrono::steady_clock::now();
                    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    cv::Vec3f mapped = surf->coord(ptr);
                    if (std::isfinite(mapped[0]) && std::isfinite(mapped[1]) && std::isfinite(mapped[2]) && !(mapped[0]==-1.f&&mapped[1]==-1.f&&mapped[2]==-1.f)) {
                        double err = cv::norm(mapped - tgt);
                        update_stats(local, dt_ms, residual, err);
                        if (residual >= 0.0f && residual <= Cfg.prof.th) local.accepted_calls++;
                        grs.push_back(residual);
                    }
                }
                // Accumulate
                g.calls += local.calls; g.time_ms += local.time_ms; g.probe_ms += local.probe_ms;
                g.residual_sum += local.residual_sum; g.err_sum += local.err_sum;
                g.residual_max = std::max(g.residual_max, local.residual_max);
                g.err_max = std::max(g.err_max, local.err_max);
                g.accepted_calls += local.accepted_calls;
            }
            // Summarize
            double combined = (g.calls? (g.time_ms + g.probe_ms)/g.calls : 0.0);
            double timeavg = (g.calls? g.time_ms/g.calls : 0.0);
            double acceptp = (g.calls? (100.0 * (double)g.accepted_calls / (double)g.calls) : 0.0);
            double ravg = (g.calls? g.residual_sum / g.calls : 0.0);
            double eavg = (g.calls? g.err_sum / g.calls : 0.0);
            double rp95=0.0; if (!grs.empty()) { std::sort(grs.begin(), grs.end()); rp95 = grs[(size_t)(0.95*(grs.size()-1))]; }
            results.push_back({Cfg.name, combined, timeavg, acceptp, rp95, ravg, g.residual_max, eavg, g.err_max});
        };

        for (auto& cfg : cfgs) run_one(cfg);

        // Find baseline and best
        auto itBase = std::find_if(results.begin(), results.end(), [](const BenchRes& r){return r.name=="baseline";});
        auto best = results[0];
        for (auto& r : results) {
            if (r.accept_percent > best.accept_percent + 1e-6 ||
               (std::abs(r.accept_percent - best.accept_percent) < 1e-6 && (r.resid_p95 < best.resid_p95 - 1e-9 ||
               (std::abs(r.resid_p95 - best.resid_p95) < 1e-9 && r.combined_avg_ms < best.combined_avg_ms)))) {
                best = r;
            }
        }

        std::cout << "\nBenchmark sweep results (accept% desc, resid_p95 asc, combined asc)\n";
        for (auto& r : results) {
            double da=0, dr=0, dc=0; if (itBase!=results.end()) { da=r.accept_percent-itBase->accept_percent; dr=r.resid_p95-itBase->resid_p95; dc=r.combined_avg_ms-itBase->combined_avg_ms; }
            std::cout << "  " << r.name << ": accept%=" << r.accept_percent
                      << " (d=" << da << ")"
                      << ", resid_p95=" << r.resid_p95 << " (d=" << dr << ")"
                      << ", combined_ms=" << r.combined_avg_ms << " (d=" << dc << ")"
                      << ", time_ms=" << r.time_avg_ms
                      << ", resid_avg=" << r.resid_avg << ", resid_max=" << r.resid_max
                      << ", error_avg=" << r.error_avg << ", error_max=" << r.error_max
                      << "\n";
        }
        if (itBase != results.end()) {
            std::cout << "Best: " << best.name << "\n"
                      << "  d accept%: " << (best.accept_percent - itBase->accept_percent) << "\n"
                      << "  d resid_p95: " << (best.resid_p95 - itBase->resid_p95) << "\n"
                      << "  d combined_ms: " << (best.combined_avg_ms - itBase->combined_avg_ms) << "\n";
        }

        // Save sweep JSON/CSV if requested
        if (save_json_path.empty()) {
            const std::string root_tag = fs::path(root_dir).filename().string();
            char buf[256];
            std::snprintf(buf, sizeof(buf), "ptto_sweep_%s_seed%u_surfs%d_samp%d.json", root_tag.c_str(), seed, (int)tifxyz_dirs.size(), samples_per_surface);
            save_json_path = buf;
            std::cout << "Auto save JSON as: " << save_json_path << std::endl;
        }

        try {
            nlohmann::json j;
            j["root"] = root_dir;
            nlohmann::json arr = nlohmann::json::array();
            for (auto& r : results) {
                nlohmann::json o;
                o["name"] = r.name;
                o["accept_percent"] = r.accept_percent;
                o["resid_p95"] = r.resid_p95;
                o["combined_avg_ms"] = r.combined_avg_ms;
                o["time_avg_ms"] = r.time_avg_ms;
                o["resid_avg"] = r.resid_avg;
                o["resid_max"] = r.resid_max;
                o["error_avg"] = r.error_avg;
                o["error_max"] = r.error_max;
                if (itBase!=results.end()) {
                    o["d_accept_vs_baseline"] = r.accept_percent - itBase->accept_percent;
                    o["d_resid_p95_vs_baseline"] = r.resid_p95 - itBase->resid_p95;
                    o["d_combined_ms_vs_baseline"] = r.combined_avg_ms - itBase->combined_avg_ms;
                }
                o["is_best"] = (r.name == best.name);
                arr.push_back(std::move(o));
            }
            j["sweep"] = arr;
            std::ofstream o(save_json_path);
            if (o) o << j.dump(2);
            std::cout << "Saved sweep JSON to " << save_json_path << std::endl;

            std::string csv = save_json_path; auto pos = csv.find_last_of('.'); if (pos!=std::string::npos) csv = csv.substr(0,pos);
            csv += ".csv";
            std::ofstream c(csv);
            if (c) {
                c << "name,accept_percent,resid_p95,combined_avg_ms,time_avg_ms,resid_avg,resid_max,error_avg,error_max,d_accept_vs_baseline,d_resid_p95_vs_baseline,d_combined_ms_vs_baseline,is_best\n";
                for (auto& r : results) {
                    double da=0, dr=0, dc=0; if (itBase!=results.end()) { da=r.accept_percent-itBase->accept_percent; dr=r.resid_p95-itBase->resid_p95; dc=r.combined_avg_ms-itBase->combined_avg_ms; }
                    c << r.name << "," << r.accept_percent << "," << r.resid_p95 << "," << r.combined_avg_ms << "," << r.time_avg_ms
                      << "," << r.resid_avg << "," << r.resid_max << "," << r.error_avg << "," << r.error_max
                      << "," << da << "," << dr << "," << dc << "," << (r.name==best.name?1:0) << "\n";
                }
            }
            std::cout << "Saved sweep CSV to " << csv << std::endl;
        } catch (...) {
            std::cerr << "Failed to save sweep outputs" << std::endl;
        }
        return 0;
    }

    for (const auto& profile : tests) {
        std::cout << "\n== Test: " << profile.name << " (init=" << profile.init
                  << ", th=" << profile.th << ", iters=" << profile.iters
                  << (coarse_probe ? ", coarse-probe:on" : ", coarse-probe:off")
                  << ") ==\n";

        Stats global_stats;
        nlohmann::json prof_json = nlohmann::json::object();
        prof_json["name"] = profile.name;
        prof_json["init"] = profile.init;
        prof_json["th"] = profile.th;
        prof_json["iters"] = profile.iters;
        prof_json["surfaces"] = nlohmann::json::array();
        // If using samples from previous JSON, align surface set to that JSON's list
        std::vector<fs::path> profile_dirs = tifxyz_dirs;
        if (!use_samples_from.empty()) {
            const nlohmann::json* prof_src = nullptr;
            if (loaded_samples.contains("profiles")) {
                for (const auto& pj : loaded_samples["profiles"]) {
                    if (pj.value("name", "") == profile.name) { prof_src = &pj; break; }
                }
                if (!prof_src && loaded_samples["profiles"].is_array() && !loaded_samples["profiles"].empty()) {
                    prof_src = &loaded_samples["profiles"][0]; // fallback to first
                }
            }
            if (prof_src) {
                std::set<std::string> want;
                if (prof_src->contains("surfaces_selected")) {
                    for (const auto& s : (*prof_src)["surfaces_selected"]) want.insert(s.get<std::string>());
                } else if (prof_src->contains("surfaces")) {
                    for (const auto& s : (*prof_src)["surfaces"]) want.insert(s.value("name", ""));
                }
                if (!want.empty()) {
                    std::map<std::string, fs::path> name2path;
                    for (const auto& p : tifxyz_dirs) name2path[p.filename().string()] = p;
                    profile_dirs.clear();
                    for (const auto& nm : want) {
                        auto it = name2path.find(nm);
                        if (it != name2path.end()) profile_dirs.push_back(it->second);
                        else if (fs::exists(fs::path(root_dir)/nm)) profile_dirs.push_back(fs::path(root_dir)/nm);
                    }
                }
            }
        }

        // Record which surfaces are selected
        nlohmann::json surfaces_list = nlohmann::json::array();
        for (auto& pth : profile_dirs) surfaces_list.push_back(pth.filename().string());
        prof_json["surfaces_selected"] = surfaces_list;

        // Collect global percentiles across all calls
        std::vector<double> gts, grs, ges; gts.reserve(profile_dirs.size()*samples_per_surface);

        for (const auto& dir : profile_dirs) {
        std::unique_ptr<QuadSurface> surf;
        try {
            surf.reset(load_quad_from_tifxyz(dir.string()));
        } catch (const std::exception& e) {
            std::cerr << "Failed to load tifxyz: " << dir << " — " << e.what() << std::endl; continue;
        }
        if (!surf) { std::cerr << "Failed to load tifxyz: " << dir << std::endl; continue; }

        cv::Mat_<cv::Vec3f>* pts = surf->rawPointsPtr();
        if (!pts || pts->empty()) { std::cerr << "Empty points in: " << dir << std::endl; continue; }

        // Pre-collect valid indices for faster sampling
        std::vector<cv::Point> valid;
        valid.reserve(pts->rows * pts->cols);
        for (int y = 1; y < pts->rows - 1; ++y) {
            for (int x = 1; x < pts->cols - 1; ++x) {
                if ((*pts)(y, x)[0] != -1.0f) valid.emplace_back(x, y);
            }
        }
        if (valid.empty()) { std::cerr << "No valid points in: " << dir << std::endl; continue; }

        std::uniform_int_distribution<size_t> uni(0, valid.size() - 1);
        std::uniform_int_distribution<int> xrand(1, pts->cols - 3);
        std::uniform_int_distribution<int> yrand(1, pts->rows - 3);

        Stats local_stats;
        size_t invalid_mappings = 0;
        size_t accepted_within_th = 0; // valid mappings with residual <= threshold
        std::vector<double> ts, rs, es; ts.reserve(samples_per_surface); rs.reserve(samples_per_surface); es.reserve(samples_per_surface);

        // Warmup a few calls to stabilize caches
        {
            cv::Vec3f ptr = surf->pointer();
            for (int w = 0; w < 10 && !valid.empty(); ++w) {
                auto p = valid[uni(rng)];
                cv::Vec3f tgt = (*pts)(p.y, p.x);
                float res = surf->pointTo(ptr, tgt, profile.th, profile.iters);
                (void)res;
            }
        }

        // Choose or load samples for this surface
        std::vector<cv::Point> samples_points;
        std::string sname = dir.filename().string();
        if (!use_samples_from.empty() && loaded_samples.contains("profiles")) {
            for (auto& pj : loaded_samples["profiles"]) {
                if (pj.value("name", "") == profile.name) {
                    for (auto& sj : pj["surfaces"]) {
                        if (sj.value("name", "") == sname && sj.contains("samples")) {
                            for (auto& sp : sj["samples"]) {
                                samples_points.emplace_back(sp[0].get<int>(), sp[1].get<int>());
                            }
                        }
                    }
                }
            }
        }
        if (samples_points.empty()) {
            samples_points.reserve(samples_per_surface);
            for (int k = 0; k < samples_per_surface; ++k) samples_points.push_back(valid[uni(rng)]);
        }

        auto bench_one_self = [&](const cv::Point& p) {
            cv::Vec3f tgt = (*pts)(p.y, p.x);

            if (noise_std > 0.0f) {
                tgt[0] += gauss(rng);
                tgt[1] += gauss(rng);
                tgt[2] += gauss(rng);
            }

            cv::Vec3f ptr;
            if (profile.init == "random") {
                // Random grid position, converted inside pointTo
                ptr = cv::Vec3f((float)xrand(rng), (float)yrand(rng), 0.0f);
            } else if (profile.init == "truth") {
                // Initialize near the true grid index for local-refine-only cost
                // Convert (x,y) grid index to internal ptr by undoing center/scale shift:
                // loc = ptr.xy + center*scale => ptr.xy = loc - center*scale
                // Here, loc ~ (x, y) on the internal absolute grid
                const cv::Vec3f loc_abs((float)p.x, (float)p.y, 0.0f);
                // center is applied as center*scale in pointTo; we can retrieve center,scale via methods
                // Estimate internal center in absolute coords as (_points->cols/2, _points->rows/2) — matches ctor
                const float cx = pts->cols/2.0f; // equals _center.x * _scale.x
                const float cy = pts->rows/2.0f; // equals _center.y * _scale.y
                ptr = cv::Vec3f(loc_abs[0] - cx, loc_abs[1] - cy, 0.0f);
            } else {
                // Default: center
                ptr = surf->pointer();
            }

            // Optional coarse probe: pick a better starting ptr via coarse sampling
            if (coarse_probe) {
                auto tpb0 = std::chrono::steady_clock::now();
                double best_rr = 1e30;
                float best_x = ptr[0] + pts->cols/2.0f;
                float best_y = ptr[1] + pts->rows/2.0f;

                if (probe_samples > 0 && !valid.empty()) {
                    std::uniform_int_distribution<size_t> pick(0, valid.size()-1);
                    for (int s = 0; s < probe_samples; ++s) {
                        auto qp = valid[pick(rng)];
                        float x = (float)qp.x;
                        float y = (float)qp.y;
                        bool ok; cv::Vec3f v = bilinear_at(*pts, x, y, ok);
                        if (!ok) continue;
                        double rr = cv::norm(v - tgt);
                        if (rr < best_rr) { best_rr = rr; best_x = x; best_y = y; }
                    }
                } else {
                    int stride = std::max(1, probe_stride);
                    for (int y = 1; y < pts->rows - 2; y += stride) {
                        for (int x = 1; x < pts->cols - 2; x += stride) {
                            bool ok; cv::Vec3f v = bilinear_at(*pts, (float)x, (float)y, ok);
                            if (!ok) continue;
                            double rr = cv::norm(v - tgt);
                            if (rr < best_rr) { best_rr = rr; best_x = (float)x; best_y = (float)y; }
                        }
                    }
                }
                // Seed ptr from best absolute loc found
                ptr[0] = best_x - pts->cols/2.0f;
                ptr[1] = best_y - pts->rows/2.0f;
                auto tpb1 = std::chrono::steady_clock::now();
                update_probe_time(local_stats, std::chrono::duration<double, std::milli>(tpb1 - tpb0).count());
            }

            auto t0 = std::chrono::steady_clock::now();
            float residual = surf->pointTo(ptr, tgt, profile.th, profile.iters);
            auto t1 = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Convert ptr to world coordinates and measure error vs target
            cv::Vec3f mapped = surf->coord(ptr);
            double err = 0.0;
            bool valid_map = !is_invalid_vec3f(mapped);
            if (!valid_map) {
                invalid_mappings++;
                // Don’t accumulate error for invalid mapping
            } else {
                err = cv::norm(mapped - tgt);
                ts.push_back(dt_ms); rs.push_back(residual); es.push_back(err);
                update_stats(local_stats, dt_ms, residual, err);
                if (residual >= 0.0 && residual <= profile.th) { accepted_within_th++; local_stats.accepted_calls++; }

                // Recompute residual from raw points at solved loc (absolute grid)
                float x_abs = ptr[0] + pts->cols/2.0f;
                float y_abs = ptr[1] + pts->rows/2.0f;
                bool ok;
                cv::Vec3f v = bilinear_at(*pts, x_abs, y_abs, ok);
                if (ok) {
                    double rr = cv::norm(v - tgt);
                    double disc = std::abs(rr - residual);
                    update_recomputed(local_stats, rr, disc);
                }
            }
        };

        if (mode == "self") {
            for (const auto& p : samples_points) bench_one_self(p);
        } else if (mode == "multisurface") {
            // In multisurface mode, we still sample targets from this surface,
            // but then we try all surfaces below, producing classification metrics later.
            for (const auto& p : samples_points) bench_one_self(p);
        }

        std::cout << "Surface: " << dir.filename().string() << "\n"
                  << "  calls: " << local_stats.calls << "\n"
                  << "  time  (avg ms): " << (local_stats.calls ? local_stats.time_ms / local_stats.calls : 0.0) << "\n"
                  << "  probe (avg ms): " << (local_stats.calls ? local_stats.probe_ms / local_stats.calls : 0.0) << "\n"
                  << "  combined (avg ms): " << (local_stats.calls ? (local_stats.time_ms + local_stats.probe_ms) / local_stats.calls : 0.0) << "\n"
                  << "  resid (avg / max): " << (local_stats.calls ? local_stats.residual_sum / local_stats.calls : 0.0) << " / " << local_stats.residual_max << "\n"
                  << "  error (avg / max): " << (local_stats.calls ? local_stats.err_sum / local_stats.calls : 0.0) << " / " << local_stats.err_max << "\n"
                  << "  invalid map (%): " << (samples_per_surface ? (100.0 * invalid_mappings / samples_per_surface) : 0.0) << "\n"
                  << "  accept@th (%): " << (local_stats.calls ? (100.0 * accepted_within_th / local_stats.calls) : 0.0) << "\n"
                  << "  recomputed (avg / max): " << (local_stats.calls ? local_stats.recomputed_sum / local_stats.calls : 0.0) << " / " << local_stats.recomputed_max << "\n"
                  << "  discrep |res-rr| (avg / max): " << (local_stats.calls ? local_stats.discrep_sum / local_stats.calls : 0.0) << " / " << local_stats.discrep_max << "\n";

        // Compute percentiles for JSON; optionally print if requested
        if (!ts.empty()) {
            double t50, t95, r50, r95, e50, e95;
            push_vals(ts, t50, t95);
            push_vals(rs, r50, r95);
            push_vals(es, e50, e95);
            if (percentiles) {
            std::cout << "  p50/p95 time (ms): " << t50 << " / " << t95 << "\n"
                      << "  p50/p95 resid    : " << r50 << " / " << r95 << "\n"
                      << "  p50/p95 error    : " << e50 << " / " << e95 << "\n";
            }
            // Append to globals
            gts.insert(gts.end(), ts.begin(), ts.end());
            grs.insert(grs.end(), rs.begin(), rs.end());
            ges.insert(ges.end(), es.begin(), es.end());
            // Add to per-surface JSON now (after s object is built below)
        }

        // Accumulate into global
        global_stats.calls += local_stats.calls;
        global_stats.time_ms += local_stats.time_ms;
        global_stats.residual_sum += local_stats.residual_sum;
        global_stats.err_sum += local_stats.err_sum;
        global_stats.residual_max = std::max(global_stats.residual_max, local_stats.residual_max);
        global_stats.err_max = std::max(global_stats.err_max, local_stats.err_max);
        global_stats.probe_ms += local_stats.probe_ms;
        global_stats.recomputed_sum += local_stats.recomputed_sum;
        global_stats.recomputed_max = std::max(global_stats.recomputed_max, local_stats.recomputed_max);
        global_stats.discrep_sum += local_stats.discrep_sum;
        global_stats.discrep_max = std::max(global_stats.discrep_max, local_stats.discrep_max);
        global_stats.accepted_calls += local_stats.accepted_calls;

        // Save JSON per surface
        nlohmann::json s;
        s["name"] = sname;
        s["calls"] = local_stats.calls;
        s["time_avg_ms"] = (local_stats.calls ? local_stats.time_ms / local_stats.calls : 0.0);
        s["probe_avg_ms"] = (local_stats.calls ? local_stats.probe_ms / local_stats.calls : 0.0);
        s["combined_avg_ms"] = (local_stats.calls ? (local_stats.time_ms + local_stats.probe_ms)/local_stats.calls : 0.0);
        s["resid_avg"] = (local_stats.calls ? local_stats.residual_sum / local_stats.calls : 0.0);
        s["resid_max"] = local_stats.residual_max;
        s["error_avg"] = (local_stats.calls ? local_stats.err_sum / local_stats.calls : 0.0);
        s["error_max"] = local_stats.err_max;
        s["recomputed_avg"] = (local_stats.calls ? local_stats.recomputed_sum / local_stats.calls : 0.0);
        s["recomputed_max"] = local_stats.recomputed_max;
        s["discrep_avg"] = (local_stats.calls ? local_stats.discrep_sum / local_stats.calls : 0.0);
        s["discrep_max"] = local_stats.discrep_max;
        s["invalid_percent"] = (samples_points.empty() ? 0.0 : (100.0 * (double)invalid_mappings / (double)samples_points.size()));
        s["accept_percent"] = (local_stats.calls ? (100.0 * (double)accepted_within_th / (double)local_stats.calls) : 0.0);
        // Per-surface percentiles
        if (!ts.empty()) {
            double t50, t95, r50, r95, e50, e95;
            push_vals(ts, t50, t95);
            push_vals(rs, r50, r95);
            push_vals(es, e50, e95);
            s["time_p50_ms"] = t50; s["time_p95_ms"] = t95;
            s["resid_p50"] = r50;   s["resid_p95"] = r95;
            s["error_p50"] = e50;   s["error_p95"] = e95;
        }
        // Save samples used
        nlohmann::json arr = nlohmann::json::array();
        for (auto& p : samples_points) {
            nlohmann::json jp = nlohmann::json::array();
            jp.push_back(p.x);
            jp.push_back(p.y);
            arr.push_back(std::move(jp));
        }
        s["samples"] = arr;
        prof_json["surfaces"].push_back(std::move(s));
    }

    if (global_stats.calls > 0) {
        std::cout << "Summary (" << profile.name << ")\n"
                  << "  calls: " << global_stats.calls << "\n"
                  << "  time  (avg ms): " << (global_stats.time_ms / global_stats.calls) << "\n"
                  << "  probe (avg ms): " << (global_stats.calls ? (global_stats.probe_ms / global_stats.calls) : 0.0) << "\n"
                  << "  combined (avg ms): " << (global_stats.calls ? (global_stats.time_ms + global_stats.probe_ms) / global_stats.calls : 0.0) << "\n"
                  << "  resid (avg / max): " << (global_stats.residual_sum / global_stats.calls) << " / " << global_stats.residual_max << "\n"
                  << "  error (avg / max): " << (global_stats.err_sum / global_stats.calls) << " / " << global_stats.err_max << "\n"
                  << "  recomputed (avg / max): " << (global_stats.recomputed_sum / global_stats.calls) << " / " << global_stats.recomputed_max << "\n"
                  << "  discrep |res-rr| (avg / max): " << (global_stats.discrep_sum / global_stats.calls) << " / " << global_stats.discrep_max << "\n"
                  << "  accept@th (%): " << (global_stats.calls ? (100.0 * (double)global_stats.accepted_calls / (double)global_stats.calls) : 0.0) << "\n";
    }

        // Save profile global to JSON
        nlohmann::json g;
        g["calls"] = global_stats.calls;
        g["time_avg_ms"] = (global_stats.calls ? global_stats.time_ms / global_stats.calls : 0.0);
        g["probe_avg_ms"] = (global_stats.calls ? global_stats.probe_ms / global_stats.calls : 0.0);
        g["combined_avg_ms"] = (global_stats.calls ? (global_stats.time_ms + global_stats.probe_ms)/global_stats.calls : 0.0);
        g["resid_avg"] = (global_stats.calls ? global_stats.residual_sum / global_stats.calls : 0.0);
        g["resid_max"] = global_stats.residual_max;
        g["error_avg"] = (global_stats.calls ? global_stats.err_sum / global_stats.calls : 0.0);
        g["error_max"] = global_stats.err_max;
        g["recomputed_avg"] = (global_stats.calls ? global_stats.recomputed_sum / global_stats.calls : 0.0);
        g["recomputed_max"] = global_stats.recomputed_max;
        g["discrep_avg"] = (global_stats.calls ? global_stats.discrep_sum / global_stats.calls : 0.0);
        g["discrep_max"] = global_stats.discrep_max;
        g["accept_percent"] = (global_stats.calls ? (100.0 * (double)global_stats.accepted_calls / (double)global_stats.calls) : 0.0);
        // Global percentiles
        if (!gts.empty()) {
            double t50, t95, r50, r95, e50, e95;
            push_vals(gts, t50, t95);
            push_vals(grs, r50, r95);
            push_vals(ges, e50, e95);
            g["time_p50_ms"] = t50; g["time_p95_ms"] = t95;
            g["resid_p50"] = r50;   g["resid_p95"] = r95;
            g["error_p50"] = e50;   g["error_p95"] = e95;
        }
        prof_json["summary"] = g;
        run_json["profiles"].push_back(std::move(prof_json));

    } // end profiles

    // Multisurface classification (optional): for each target from each surface,
    // run pointTo against all surfaces and report top-1 and recall@threshold.
    if (mode == "multisurface") {
        std::cout << "\nMultisurface membership check (top-1 / recall@th / FP rate)\n";
        // Load all surfaces
        struct Entry { fs::path dir; std::unique_ptr<QuadSurface> surf; cv::Mat_<cv::Vec3f>* pts; std::vector<cv::Point> valid; };
        std::vector<Entry> entries;
        entries.reserve(tifxyz_dirs.size());
        for (const auto& d : tifxyz_dirs) {
            try {
                std::unique_ptr<QuadSurface> s(load_quad_from_tifxyz(d.string()));
                if (!s) continue;
                auto* p = s->rawPointsPtr();
                if (!p || p->empty()) continue;
                Entry e{d, std::move(s), p, {}};
                for (int y = 1; y < p->rows - 1; ++y) for (int x = 1; x < p->cols - 1; ++x) if ((*p)(y,x)[0] != -1.0f) e.valid.emplace_back(x,y);
                if (!e.valid.empty()) entries.push_back(std::move(e));
            } catch (...) {}
        }
        if (entries.size() >= 2) {
            std::mt19937 rng2(seed ? seed : (unsigned)std::random_device{}());
            std::uniform_int_distribution<size_t> pick;
            size_t total = 0, top1 = 0, recall = 0; double fp_sum = 0.0;

            for (auto& E : entries) {
                if (E.valid.empty()) continue;
                pick = std::uniform_int_distribution<size_t>(0, E.valid.size()-1);
                for (int k = 0; k < samples_per_surface; ++k) {
                    auto p = E.valid[pick(rng2)];
                    cv::Vec3f tgt = (*E.pts)(p.y, p.x);
                    total++;

                    double best_res = 1e30; int best_idx = -1; int true_idx = -1; int fps = 0; double true_res = 1e30;
                    for (int j = 0; j < (int)entries.size(); ++j) {
                        auto& F = entries[j];
                        cv::Vec3f ptr = F.surf->pointer();
                        float r = F.surf->pointTo(ptr, tgt, threshold, max_iters);
                        if (r >= 0 && r < best_res) { best_res = r; best_idx = j; }
                        if (&F == &E) { true_idx = j; true_res = r; }
                        if (r >= 0 && r <= threshold && &F != &E) fps++;
                    }
                    if (best_idx == true_idx) top1++;
                    if (true_res >= 0 && true_res <= threshold) recall++;
                    fp_sum += fps;
                }
            }
            if (total > 0) {
                std::cout << "  samples: " << total << "\n"
                          << "  top1 acc: " << (100.0 * top1 / total) << "%\n"
                          << "  recall@th: " << (100.0 * recall / total) << "%\n"
                          << "  FP rate: " << (fp_sum / total) << " (avg other surfaces passing threshold)\n";
            }
        } else {
            std::cout << "  Not enough surfaces for multisurface mode." << std::endl;
        }
    }

    if (!save_json_path.empty()) {
        try {
            std::ofstream o(save_json_path);
            if (o) o << run_json.dump(2);
            std::cout << "Saved metrics JSON to " << save_json_path << std::endl;
        } catch (...) {
            std::cerr << "Failed to save JSON to " << save_json_path << std::endl;
        }
    }

    // CSV export: mirror core fields per surface and summary
    std::string csv_out;
    if (!save_json_path.empty()) {
        csv_out = save_json_path;
        auto pos = csv_out.find_last_of('.');
        if (pos != std::string::npos) csv_out = csv_out.substr(0, pos);
        csv_out += ".csv";
    }
    if (!csv_out.empty()) {
        try {
            std::ofstream c(csv_out);
            if (c) {
                c << "profile,surface,calls,time_avg_ms,probe_avg_ms,combined_avg_ms,resid_avg,resid_max,error_avg,error_max,recomputed_avg,recomputed_max,discrep_avg,discrep_max,invalid_percent,accept_percent,time_p50_ms,time_p95_ms,resid_p50,resid_p95,error_p50,error_p95\n";
                for (auto& pj : run_json["profiles"]) {
                    const std::string pname = pj.value("name", "");
                    for (auto& s : pj["surfaces"]) {
                        c << pname << "," << s.value("name","") << ","
                          << s.value("calls",0) << ","
                          << s.value("time_avg_ms",0.0) << ","
                          << s.value("probe_avg_ms",0.0) << ","
                          << s.value("combined_avg_ms",0.0) << ","
                          << s.value("resid_avg",0.0) << ","
                          << s.value("resid_max",0.0) << ","
                          << s.value("error_avg",0.0) << ","
                          << s.value("error_max",0.0) << ","
                          << s.value("recomputed_avg",0.0) << ","
                          << s.value("recomputed_max",0.0) << ","
                          << s.value("discrep_avg",0.0) << ","
                          << s.value("discrep_max",0.0) << ","
                          << s.value("invalid_percent",0.0) << ","
                          << s.value("accept_percent",0.0) << ","
                          << s.value("time_p50_ms",0.0) << ","
                          << s.value("time_p95_ms",0.0) << ","
                          << s.value("resid_p50",0.0) << ","
                          << s.value("resid_p95",0.0) << ","
                          << s.value("error_p50",0.0) << ","
                          << s.value("error_p95",0.0) << "\n";
                    }
                    // Summary row
                    auto g = pj["summary"];
                    c << pname << ",SUMMARY,"
                      << g.value("calls",0) << ","
                      << g.value("time_avg_ms",0.0) << ","
                      << g.value("probe_avg_ms",0.0) << ","
                      << g.value("combined_avg_ms",0.0) << ","
                      << g.value("resid_avg",0.0) << ","
                      << g.value("resid_max",0.0) << ","
                      << g.value("error_avg",0.0) << ","
                      << g.value("error_max",0.0) << ","
                      << g.value("recomputed_avg",0.0) << ","
                      << g.value("recomputed_max",0.0) << ","
                      << g.value("discrep_avg",0.0) << ","
                      << g.value("discrep_max",0.0) << ","
                      << "" << ","  // invalid_percent not tracked globally
                      << g.value("accept_percent",0.0) << ","
                      << g.value("time_p50_ms",0.0) << ","
                      << g.value("time_p95_ms",0.0) << ","
                      << g.value("resid_p50",0.0) << ","
                      << g.value("resid_p95",0.0) << ","
                      << g.value("error_p50",0.0) << ","
                      << g.value("error_p95",0.0) << "\n";
                }
            }
            std::cout << "Saved metrics CSV to " << csv_out << std::endl;
        } catch (...) {
            std::cerr << "Failed to save CSV to " << csv_out << std::endl;
        }
    }

    // Compare to previous JSON if requested
    if (!compare_to_path.empty()) {
        try {
            nlohmann::json prev;
            std::ifstream in(compare_to_path);
            if (in) in >> prev;
            std::cout << "\nComparison vs " << compare_to_path << "\n";
            auto show = [&](const std::string& prof_name){
                auto curp = std::find_if(run_json["profiles"].begin(), run_json["profiles"].end(), [&](const nlohmann::json& j){return j.value("name","")==prof_name;});
                auto prvp = std::find_if(prev["profiles"].begin(), prev["profiles"].end(), [&](const nlohmann::json& j){return j.value("name","")==prof_name;});
                if (curp==run_json["profiles"].end() || prvp==prev["profiles"].end()) return;
                std::cout << "Profile: " << prof_name << "\n";
                // Global deltas
                auto cg = (*curp)["summary"], pg = (*prvp)["summary"];
                auto d = [&](const char* k){ return cg.value(k,0.0) - pg.value(k,0.0); };
                std::cout << "  d time_avg_ms: " << d("time_avg_ms") << ", d probe_avg_ms: " << d("probe_avg_ms")
                          << ", d combined_avg_ms: " << d("combined_avg_ms") << "\n"
                          << "  d resid_avg: " << d("resid_avg") << ", d error_avg: " << d("error_avg") << ", d accept%: " << d("accept_percent") << "\n"
                          << "  d time_p50_ms: " << d("time_p50_ms") << ", d time_p95_ms: " << d("time_p95_ms") << "\n"
                          << "  d resid_p50: " << d("resid_p50") << ", d resid_p95: " << d("resid_p95") << "\n"
                          << "  d error_p50: " << d("error_p50") << ", d error_p95: " << d("error_p95") << "\n";
                // Per surface deltas (matching by name)
                for (auto& cs : (*curp)["surfaces"]) {
                    auto name = cs.value("name","");
                    auto it = std::find_if((*prvp)["surfaces"].begin(), (*prvp)["surfaces"].end(), [&](const nlohmann::json& s){return s.value("name","")==name;});
                    if (it==(*prvp)["surfaces"].end()) continue;
                    std::cout << "  " << name << ": d combined_ms=" << cs.value("combined_avg_ms",0.0)-(*it).value("combined_avg_ms",0.0)
                              << ", d resid_avg=" << cs.value("resid_avg",0.0)-(*it).value("resid_avg",0.0)
                              << ", d error_avg=" << cs.value("error_avg",0.0)-(*it).value("error_avg",0.0)
                              << ", d accept%=" << cs.value("accept_percent",0.0)-(*it).value("accept_percent",0.0)
                              << ", d invalid%=" << cs.value("invalid_percent",0.0)-(*it).value("invalid_percent",0.0)
                              << ", d time_p95_ms=" << cs.value("time_p95_ms",0.0)-(*it).value("time_p95_ms",0.0)
                              << ", d resid_p95=" << cs.value("resid_p95",0.0)-(*it).value("resid_p95",0.0)
                              << ", d error_p95=" << cs.value("error_p95",0.0)-(*it).value("error_p95",0.0)
                              << "\n";
                }
            };
            for (auto& p : run_json["profiles"]) show(p.value("name",""));
        } catch (...) {
            std::cerr << "Failed to compare against JSON: " << compare_to_path << std::endl;
        }
    }

    return 0;
}
