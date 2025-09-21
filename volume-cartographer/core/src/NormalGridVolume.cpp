#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/HashFunctions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace vc::core::util {

    struct NormalGridVolume::pimpl {
        struct CacheEntry {
            std::shared_ptr<GridStore> grid;
            bool missing = false;
            std::list<cv::Vec2i>::iterator lru_it{};
            bool has_lru = false;
        };

        static size_t resolve_cache_limit(const nlohmann::json& metadata) {
            constexpr size_t kDefaultMaxOpenGrids = 50000;
            size_t limit = kDefaultMaxOpenGrids;

            const char* keys[] = {"normal-grid-cache-limit", "normal_grid_cache_limit"};
            for (const char* key : keys) {
                auto it = metadata.find(key);
                if (it != metadata.end()) {
                    try {
                        auto value = it->get<size_t>();
                        if (value > 0) {
                            limit = value;
                            break;
                        }
                    } catch (const std::exception&) {
                        // Ignore invalid metadata override and fall back to defaults.
                    }
                }
            }

            if (const char* env_limit = std::getenv("NORMAL_GRID_VOLUME_CACHE_LIMIT")) {
                char* endptr = nullptr;
                unsigned long long parsed = std::strtoull(env_limit, &endptr, 10);
                if (endptr != env_limit && *endptr == '\0' && parsed > 0) {
                    limit = static_cast<size_t>(parsed);
                }
            }

            return std::max<size_t>(limit, 1);
        }

        std::string base_path;
        int sparse_volume;
        nlohmann::json metadata;
        size_t max_cache_entries;
        std::vector<std::string> plane_dirs = {"xy", "xz", "yz"};

        mutable std::mutex mutex;
        mutable std::unordered_map<cv::Vec2i, CacheEntry> grid_cache;
        mutable std::list<cv::Vec2i> lru_list;

        explicit pimpl(const std::string& path) : base_path(path) {
            std::ifstream metadata_file((fs::path(base_path) / "metadata.json").string());
            if (!metadata_file.is_open()) {
                throw std::runtime_error("Failed to open metadata.json in " + base_path);
            }
            metadata_file >> metadata;
            sparse_volume = metadata.value("sparse-volume", 1);
            max_cache_entries = resolve_cache_limit(metadata);
        }

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const {
            float coord;
            switch (plane_idx) {
                case 0: coord = point.z; break; // XY plane
                case 1: coord = point.y; break; // XZ plane
                case 2: coord = point.x; break; // YZ plane
                default: return std::nullopt;
            }

            int slice_idx1 = static_cast<int>(coord / sparse_volume) * sparse_volume;
            int slice_idx2 = slice_idx1 + sparse_volume;

            double weight = (coord - slice_idx1) / sparse_volume;

            auto grid1 = get_grid(plane_idx, slice_idx1);
            auto grid2 = get_grid(plane_idx, slice_idx2);

            if (!grid1 || !grid2) {
                return std::nullopt;
            }

            return GridQueryResult{std::move(grid1), std::move(grid2), weight};
        }

        std::shared_ptr<const GridStore> query_nearest(const cv::Point3f& point, int plane_idx) const {
            float coord;
            switch (plane_idx) {
                case 0: coord = point.z; break; // XY plane
                case 1: coord = point.y; break; // XZ plane
                case 2: coord = point.x; break; // YZ plane
                default: return nullptr;
            }

            int slice_idx = static_cast<int>(std::round(coord / sparse_volume)) * sparse_volume;

            return get_grid(plane_idx, slice_idx);
        }

        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx) const {
            if (plane_idx < 0 || plane_idx >= static_cast<int>(plane_dirs.size())) {
                return nullptr;
            }

            cv::Vec2i key(plane_idx, slice_idx);

            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = grid_cache.find(key);
                if (it != grid_cache.end()) {
                    auto& entry = it->second;
                    if (entry.missing) {
                        return nullptr;
                    }
                    if (entry.grid) {
                        touch_entry_locked(entry, key);
                        return entry.grid;
                    }
                }
            }

            const std::string& dir = plane_dirs[plane_idx];
            char filename[256];
            snprintf(filename, sizeof(filename), "%06d.grid", slice_idx);
            std::string grid_path = (fs::path(base_path) / dir / filename).string();

            std::shared_ptr<GridStore> loaded_grid;
            bool missing = false;

            if (!fs::exists(grid_path)) {
                missing = true;
            } else {
                loaded_grid = std::make_shared<GridStore>(grid_path);
            }

            std::lock_guard<std::mutex> lock(mutex);
            auto [it, inserted] = grid_cache.try_emplace(key);
            auto& entry = it->second;

            if (!inserted) {
                if (entry.missing) {
                    return nullptr;
                }
                if (entry.grid) {
                    touch_entry_locked(entry, key);
                    return entry.grid;
                }
            }

            if (missing) {
                entry.grid.reset();
                entry.missing = true;
                if (entry.has_lru) {
                    lru_list.erase(entry.lru_it);
                    entry.has_lru = false;
                    entry.lru_it = {};
                }
                return nullptr;
            }

            entry.grid = std::move(loaded_grid);
            entry.missing = false;
            touch_entry_locked(entry, key);
            enforce_cache_limit_locked();

            return entry.grid;
        }

        void touch_entry_locked(CacheEntry& entry, const cv::Vec2i& key) const {
            if (entry.missing) {
                if (entry.has_lru) {
                    lru_list.erase(entry.lru_it);
                    entry.has_lru = false;
                    entry.lru_it = {};
                }
                return;
            }

            if (entry.has_lru) {
                lru_list.splice(lru_list.begin(), lru_list, entry.lru_it);
            } else {
                lru_list.push_front(key);
                entry.lru_it = lru_list.begin();
                entry.has_lru = true;
            }
        }

        void enforce_cache_limit_locked() const {
            while (lru_list.size() > max_cache_entries) {
                const cv::Vec2i key_to_evict = lru_list.back();
                lru_list.pop_back();

                auto it = grid_cache.find(key_to_evict);
                if (it != grid_cache.end()) {
                    it->second.has_lru = false;
                    grid_cache.erase(it);
                }
            }
        }
    };

    NormalGridVolume::NormalGridVolume(const std::string& path)
        : pimpl_(std::make_unique<pimpl>(path)) {}

    std::optional<NormalGridVolume::GridQueryResult> NormalGridVolume::query(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query(point, plane_idx);
    }

    std::shared_ptr<const GridStore> NormalGridVolume::query_nearest(const cv::Point3f& point, int plane_idx) const {
        return pimpl_->query_nearest(point, plane_idx);
    }

    NormalGridVolume::~NormalGridVolume() = default;
    NormalGridVolume::NormalGridVolume(NormalGridVolume&&) noexcept = default;
    NormalGridVolume& NormalGridVolume::operator=(NormalGridVolume&&) noexcept = default;
    const nlohmann::json& NormalGridVolume::metadata() const {
        return pimpl_->metadata;
    }
} // namespace vc::core::util
