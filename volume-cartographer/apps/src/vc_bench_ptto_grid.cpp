// Baseline benchmark for grid-based free-function pointTo over raw points
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
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

static inline bool bad(const cv::Vec3f& v){ return !std::isfinite(v[0])||!std::isfinite(v[1])||!std::isfinite(v[2])||(v[0]==-1.f&&v[1]==-1.f&&v[2]==-1.f); }

// Bilinear sampling of a Vec3f image at floating coords; returns ok=false if any neighbor is invalid
static cv::Vec3f bilinear_at(const cv::Mat_<cv::Vec3f>& m, float x, float y, bool& ok) {
    ok = false;
    if (m.empty()) return {NAN,NAN,NAN};
    if (x < 0.0f || y < 0.0f || x > (float)(m.cols - 2) || y > (float)(m.rows - 2)) return {NAN,NAN,NAN};
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    float dx = x - x0, dy = y - y0;
    auto p00 = m(y0, x0), p10 = m(y0, x0+1), p01 = m(y0+1, x0), p11 = m(y0+1, x0+1);
    if (bad(p00) || bad(p10) || bad(p01) || bad(p11)) return {NAN,NAN,NAN};
    cv::Vec3f a = p00*(1.0f-dx) + p10*dx;
    cv::Vec3f b = p01*(1.0f-dx) + p11*dx;
    ok = true;
    return a*(1.0f-dy) + b*dy;
}

struct Stats {
    size_t calls = 0;
    double time_ms = 0.0;
    double resid_sum = 0.0;
    double resid_max = 0.0;
    double err_sum = 0.0;
    double err_max = 0.0;
    size_t accepted = 0;
};

int main(int argc, char** argv) {
    std::string root_dir;
    int samples_per_surface = 2000;
    int surfaces_limit = 0; // 0 = all
    unsigned int seed = 0;
    float threshold = 2.0f;
    int max_iters = 10;
    int roi_size = 0; // 0 = full surface
    float scale_override = 0.0f; // 0 = use surf->scale()[0]
    std::string save_csv_path;
    std::string use_samples_from;
    std::string save_samples_path;

    po::options_description desc("Baseline benchmark: grid pointTo (free-function)");
    desc.add_options()
        ("help,h", "Print help")
        ("root,r", po::value<std::string>(&root_dir)->required(), "Root directory with tifxyz subdirs")
        ("samples,s", po::value<int>(&samples_per_surface)->default_value(samples_per_surface), "Samples per surface")
        ("surfaces", po::value<int>(&surfaces_limit)->default_value(surfaces_limit), "Max number of surfaces (0 = all)")
        ("seed", po::value<unsigned int>(&seed)->default_value(seed), "RNG seed (0 = nondeterministic)")
        ("threshold,t", po::value<float>(&threshold)->default_value(threshold), "Residual acceptance threshold")
        ("iters,i", po::value<int>(&max_iters)->default_value(max_iters), "Max iterations for pointTo")
        ("roi", po::value<int>(&roi_size)->default_value(roi_size), "Square ROI size (0 = full surface)")
        ("scale", po::value<float>(&scale_override)->default_value(scale_override), "Scale argument for grid pointTo (0 = use surf->scale()[0])")
        ("save-csv", po::value<std::string>(&save_csv_path), "Save per-surface metrics CSV")
        ("use-samples-from", po::value<std::string>(&use_samples_from), "Load fixed samples JSON from previous run")
        ("save-samples", po::value<std::string>(&save_samples_path), "Save chosen samples to JSON for reuse");

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
    Stats global;
    struct SurfRow { std::string name; Stats s; };
    std::vector<SurfRow> rows;
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
        float scale = (scale_override != 0.0f ? scale_override : surf->scale()[0]);
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

        // Warmup
        {
            cv::Vec2f loc;
            for (int w = 0; w < 8 && !valid.empty(); ++w) {
                auto p = valid[pick(rng)];
                (void)pointTo(loc, *pts, (*pts)(p.y,p.x), threshold, max_iters, scale);
            }
        }

        Stats local;
        for (int k = 0; k < (int)samples_points.size(); ++k) {
            auto p = samples_points[k];
            cv::Vec3f tgt = (*pts)(p.y, p.x);

            const cv::Mat_<cv::Vec3f>* mat = pts;
            cv::Mat_<cv::Vec3f> roi;
            int xoff = 0, yoff = 0;
            if (roi_size > 0) {
                int RS = std::max(8, roi_size);
                RS = std::min(RS, std::min(pts->cols, pts->rows));
                int x0 = std::max(0, std::min(p.x - RS/2, pts->cols - RS));
                int y0 = std::max(0, std::min(p.y - RS/2, pts->rows - RS));
                if (RS < 3 || x0 < 0 || y0 < 0 || x0 + RS > pts->cols || y0 + RS > pts->rows) continue;
                roi = (*pts)(cv::Rect(x0, y0, RS, RS));
                mat = &roi; xoff = x0; yoff = y0;
            }

            auto t0 = std::chrono::steady_clock::now();
            cv::Vec2f loc;
            float residual = pointTo(loc, *mat, tgt, threshold, max_iters, scale);
            auto t1 = std::chrono::steady_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            bool ok = false; cv::Vec3f mapped = bilinear_at(*mat, loc[0], loc[1], ok);
            if (!ok) continue;

            double err = cv::norm(mapped - tgt);
            local.calls++;
            local.time_ms += dt_ms;
            local.resid_sum += residual;
            local.resid_max = std::max(local.resid_max, (double)residual);
            local.err_sum += err;
            local.err_max = std::max(local.err_max, err);
            if (residual >= 0.0f && residual <= threshold) local.accepted++;
        }

        rows.push_back({sname, local});
        global.calls += local.calls;
        global.time_ms += local.time_ms;
        global.resid_sum += local.resid_sum;
        global.resid_max = std::max(global.resid_max, local.resid_max);
        global.err_sum += local.err_sum;
        global.err_max = std::max(global.err_max, local.err_max);
        global.accepted += local.accepted;
    }

    if (global.calls == 0) { std::cerr << "No valid samples processed." << std::endl; return 1; }

    std::cout << "Grid pointTo baseline\n";
    std::cout << "surfaces: " << dirs.size() << ", calls: " << global.calls << "\n";
    std::cout << "time_avg_ms: " << (global.time_ms / global.calls) << "\n";
    std::cout << "resid_avg: " << (global.resid_sum / global.calls) << ", resid_max: " << global.resid_max << "\n";
    std::cout << "error_avg: " << (global.err_sum / global.calls) << ", error_max: " << global.err_max << "\n";
    std::cout << "accept_percent: " << (100.0 * (double)global.accepted / (double)global.calls) << "\n";
    if (!save_csv_path.empty()) {
        try {
            std::ofstream c(save_csv_path);
            if (c) {
                c << "surface,calls,time_avg_ms,resid_avg,resid_max,error_avg,error_max,accept_percent\n";
                for (const auto& r : rows) {
                    const Stats& s = r.s;
                    double tavg = (s.calls? s.time_ms / s.calls : 0.0);
                    double ravg = (s.calls? s.resid_sum / s.calls : 0.0);
                    double eavg = (s.calls? s.err_sum / s.calls : 0.0);
                    double apct = (s.calls? (100.0 * (double)s.accepted / (double)s.calls) : 0.0);
                    c << r.name << "," << s.calls << "," << tavg << "," << ravg << "," << s.resid_max
                      << "," << eavg << "," << s.err_max << "," << apct << "\n";
                }
                double tavg = (global.calls? global.time_ms / global.calls : 0.0);
                double ravg = (global.calls? global.resid_sum / global.calls : 0.0);
                double eavg = (global.calls? global.err_sum / global.calls : 0.0);
                double apct = (global.calls? (100.0 * (double)global.accepted / (double)global.calls) : 0.0);
                c << "SUMMARY," << global.calls << "," << tavg << "," << ravg << "," << global.resid_max
                  << "," << eavg << "," << global.err_max << "," << apct << "\n";
            }
        } catch (...) { std::cerr << "Failed to save CSV: " << save_csv_path << std::endl; }
    }
    if (!save_samples_path.empty()) {
        try { std::ofstream o(save_samples_path); if (o) o << samples_json.dump(2); } catch (...) { std::cerr << "Failed to save samples JSON: " << save_samples_path << std::endl; }
    }
    return 0;
}
