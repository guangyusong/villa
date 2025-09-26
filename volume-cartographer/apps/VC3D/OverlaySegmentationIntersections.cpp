#include "OverlaySegmentationIntersections.hpp"

#include "CVolumeViewer.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <QGraphicsScene>
#include <QGraphicsPathItem>
#include <QObject>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <omp.h>

namespace {
constexpr auto kColorSegYZ = Qt::yellow;
constexpr auto kColorSegXZ = Qt::red;
const QColor kColorSegXY(255, 140, 0);
} // namespace

OverlaySegmentationIntersections::OverlaySegmentationIntersections(CSurfaceCollection& surfaces)
    : _surfaces(surfaces)
{
}

OverlaySegmentationIntersections::~OverlaySegmentationIntersections()
{
    for (auto& [viewer, state] : _viewers) {
        if (viewer) {
            clearAllItems(*viewer, state);
            viewer->setSegmentationOverlay(nullptr);
        }
    }
}

void OverlaySegmentationIntersections::registerViewer(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    viewer->setSegmentationOverlay(this);
    auto& state = ensureState(*viewer);
    if (state.targets.empty()) {
        state.targets.insert("visible_segmentation");
    }

    QObject::connect(viewer, &QObject::destroyed, [this](QObject* obj) {
        if (auto* died = qobject_cast<CVolumeViewer*>(obj)) {
            auto it = _viewers.find(died);
            if (it != _viewers.end()) {
                clearAllItems(*died, it->second);
                _viewers.erase(it);
            }
        }
    });

    refreshViewer(*viewer);
}

void OverlaySegmentationIntersections::unregisterViewer(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    auto it = _viewers.find(viewer);
    if (it != _viewers.end()) {
        clearAllItems(*viewer, it->second);
        _viewers.erase(it);
    }
    viewer->setSegmentationOverlay(nullptr);
}

std::set<std::string> OverlaySegmentationIntersections::intersectsForViewer(const CVolumeViewer& viewer) const
{
    if (const auto* state = lookupState(viewer)) {
        return state->targets;
    }
    return {};
}

void OverlaySegmentationIntersections::setIntersectsForViewer(CVolumeViewer& viewer,
                                                              const std::set<std::string>& targets)
{
    auto& state = ensureState(viewer);
    if (state.targets == targets) {
        return;
    }
    state.targets = targets;
    invalidateViewer(viewer);
    refreshViewer(viewer);
}

void OverlaySegmentationIntersections::handleIntersectionChanged(CVolumeViewer& viewer,
                                                                  const std::string& a,
                                                                  const std::string& b,
                                                                  Intersection* intersection)
{
    auto& state = ensureState(viewer);
    if (state.ignoreIntersectionChange && intersection == state.ignoreIntersectionChange) {
        return;
    }

    if (!state.targets.count(a) || !state.targets.count(b)) {
        return;
    }

    const std::string& surfName = viewer.surfName();
    if (a == surfName || (surfName == "segmentation" && a == "visible_segmentation")) {
        invalidateViewer(viewer, b);
    } else if (b == surfName || (surfName == "segmentation" && b == "visible_segmentation")) {
        invalidateViewer(viewer, a);
    }

    refreshViewer(viewer);
}

void OverlaySegmentationIntersections::invalidateViewer(CVolumeViewer& viewer,
                                                         const std::optional<std::string>& key)
{
    auto it = _viewers.find(&viewer);
    if (it == _viewers.end()) {
        return;
    }

    if (!key.has_value()) {
        clearAllItems(viewer, it->second);
        return;
    }

    clearItems(viewer, it->second, *key);
}

void OverlaySegmentationIntersections::hideViewerItems(CVolumeViewer& viewer)
{
    auto it = _viewers.find(&viewer);
    if (it == _viewers.end()) {
        return;
    }
    for (auto& [_, items] : it->second.items) {
        for (auto* item : items) {
            if (item) {
                item->setVisible(false);
            }
        }
    }
}

void OverlaySegmentationIntersections::clear()
{
    for (auto& [viewer, state] : _viewers) {
        if (!viewer) {
            continue;
        }
        clearAllItems(*viewer, state);
        state.items.clear();
        state.ignoreIntersectionChange = nullptr;
    }
}

void OverlaySegmentationIntersections::refreshViewer(CVolumeViewer& viewer)
{
    const auto volume = viewer.currentVolume();
    if (!volume || !volume->zarrDataset()) {
        return;
    }

    Surface* surface = viewer.surface();
    if (!surface) {
        return;
    }

    auto* scene = viewer.scene();
    if (!scene) {
        return;
    }

    auto& state = ensureState(viewer);

    // Drop overlays for targets that are no longer requested
    {
        std::vector<std::string> stale;
        stale.reserve(state.items.size());
        for (const auto& [key, _] : state.items) {
            if (!state.targets.count(key)) {
                stale.push_back(key);
            }
        }
        for (const auto& key : stale) {
            clearItems(viewer, state, key);
        }
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(surface)) {
        const float scale = std::max(viewer.scale(), 1e-6f);
        const QRect imgArea = viewer.currentImageArea();
        const cv::Rect planeRoi(
            static_cast<int>(std::floor(imgArea.x() / scale)),
            static_cast<int>(std::floor(imgArea.y() / scale)),
            static_cast<int>(std::ceil(imgArea.width() / scale)),
            static_cast<int>(std::ceil(imgArea.height() / scale))
        );

        cv::Vec3f corner = plane->coord(cv::Vec3f(0.f, 0.f, 0.f), {static_cast<float>(planeRoi.x), static_cast<float>(planeRoi.y), 0.f});
        Rect3D viewBBox{corner, corner};
        viewBBox = expand_rect(viewBBox, plane->coord(cv::Vec3f(0.f, 0.f, 0.f), {static_cast<float>(planeRoi.br().x), static_cast<float>(planeRoi.y), 0.f}));
        viewBBox = expand_rect(viewBBox, plane->coord(cv::Vec3f(0.f, 0.f, 0.f), {static_cast<float>(planeRoi.x), static_cast<float>(planeRoi.br().y), 0.f}));
        viewBBox = expand_rect(viewBBox, plane->coord(cv::Vec3f(0.f, 0.f, 0.f), {static_cast<float>(planeRoi.br().x), static_cast<float>(planeRoi.br().y), 0.f}));

        std::vector<std::string> targets(state.targets.begin(), state.targets.end());
        std::vector<std::string> candidates;
        candidates.reserve(targets.size());

#pragma omp parallel for
        for (int n = 0; n < static_cast<int>(targets.size()); ++n) {
            const std::string& key = targets[static_cast<std::size_t>(n)];
            bool already;
#pragma omp critical
            already = state.items.count(key);
            if (!already) {
                if (auto* segmentation = dynamic_cast<QuadSurface*>(_surfaces.surface(key))) {
                    if (intersect(viewBBox, segmentation->bbox())) {
#pragma omp critical
                        candidates.push_back(key);
                    } else {
#pragma omp critical
                        {
                            state.items[key] = {};
                        }
                    }
                }
            }
        }

        std::vector<std::vector<std::vector<cv::Vec3f>>> intersections(candidates.size());

#pragma omp parallel for
        for (int idx = 0; idx < static_cast<int>(candidates.size()); ++idx) {
            const std::string& key = candidates[static_cast<std::size_t>(idx)];
            auto* segmentation = dynamic_cast<QuadSurface*>(_surfaces.surface(key));
            if (!segmentation) {
                continue;
            }
            std::vector<std::vector<cv::Vec2f>> unusedGrid;
            const float step = 4.f / scale;
            const int minTries = (key == "segmentation") ? 1000 : 10;
            find_intersect_segments(intersections[static_cast<std::size_t>(idx)],
                                    unusedGrid,
                                    segmentation->rawPoints(),
                                    plane,
                                    planeRoi,
                                    step,
                                    minTries);
        }

        std::hash<std::string> hasher;

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const std::string& key = candidates[i];
            const auto& segIntersections = intersections[i];

            clearItems(viewer, state, key);

            if (segIntersections.empty()) {
                continue;
            }

            QColor color;
            float width = 2.f;
            int zValue = 5;

            if (key == "segmentation") {
                const auto& surfName = viewer.surfName();
                color = (surfName == "seg yz" ? kColorSegYZ
                         : surfName == "seg xz" ? kColorSegXZ
                                                 : kColorSegXY);
                width = 3.f;
                zValue = 20;
            } else {
                const auto seed = static_cast<unsigned int>(hasher(key));
                std::srand(seed);
                cv::Vec3i cvColor = {
                    100 + std::rand() % 255,
                    100 + std::rand() % 255,
                    100 + std::rand() % 255
                };
                const int prim = std::rand() % 3;
                cvColor[prim] = 200 + std::rand() % 55;
                color = QColor(std::clamp(cvColor[0], 0, 255),
                               std::clamp(cvColor[1], 0, 255),
                               std::clamp(cvColor[2], 0, 255));
            }

            const auto paths = buildPathsFromSegments(segIntersections, *plane, scale);
            std::vector<QGraphicsItem*> items;
            items.reserve(paths.size());

            QPen pen(color, width);
            for (const auto& path : paths) {
                auto* item = scene->addPath(path, pen);
                item->setZValue(zValue);
                items.push_back(item);
            }

            state.items[key] = items;
            state.ignoreIntersectionChange = new Intersection{segIntersections};
            _surfaces.setIntersection(viewer.surfName(), key, state.ignoreIntersectionChange);
            state.ignoreIntersectionChange = nullptr;
        }
    } else if (viewer.surfName() == "segmentation") {
        const auto intersects = _surfaces.intersections("segmentation");
        for (const auto& pair : intersects) {
            std::string key = pair.first;
            if (key == "segmentation") {
                key = pair.second;
            }

            if (state.items.count(key) || !state.targets.count(key)) {
                continue;
            }

            auto* segmentationSurface = dynamic_cast<QuadSurface*>(viewer.surface());
            if (!segmentationSurface) {
                continue;
            }

            std::unordered_map<cv::Vec3f, cv::Vec3f, vec3f_hash> locationCache;
            std::vector<cv::Vec3f> srcLocations;

            if (auto* intersection = _surfaces.intersection(pair.first, pair.second)) {
                for (const auto& seg : intersection->lines) {
                    srcLocations.insert(srcLocations.end(), seg.begin(), seg.end());
                }
            }

#pragma omp parallel
            {
                auto ptr = segmentationSurface->pointer();
#pragma omp for
                for (std::size_t idx = 0; idx < srcLocations.size(); ++idx) {
                    const auto wp = srcLocations[idx];
                    float res = segmentationSurface->pointTo(ptr, wp, 2.0f, 100);
                    cv::Vec3f loc = segmentationSurface->loc(ptr) * viewer.scale();
                    if (res >= 2.0f) {
                        loc = {-1.f, -1.f, -1.f};
                    }
#pragma omp critical
                    locationCache[wp] = loc;
                }
            }

            std::vector<QGraphicsItem*> items;
            if (auto* intersection = _surfaces.intersection(pair.first, pair.second)) {
                for (const auto& seg : intersection->lines) {
                    QPainterPath path;
                    bool first = true;
                    cv::Vec3f last = {-1.f, -1.f, -1.f};
                    for (const auto& wp : seg) {
                        const auto loc = locationCache[wp];
                        if (loc[0] == -1.f) {
                            continue;
                        }
                        if (last[0] != -1.f && cv::norm(loc - last) >= 8.f) {
                            if (auto* scene = viewer.scene()) {
                                auto* item = scene->addPath(path, QPen(key == "seg yz" ? kColorSegYZ : kColorSegXZ, 2));
                                item->setZValue(5);
                                items.push_back(item);
                            }
                            path = QPainterPath();
                            first = true;
                        }
                        last = loc;
                        if (first) {
                            path.moveTo(loc[0], loc[1]);
                        } else {
                            path.lineTo(loc[0], loc[1]);
                        }
                        first = false;
                    }
                    if (auto* scene = viewer.scene()) {
                        auto* item = scene->addPath(path, QPen(key == "seg yz" ? kColorSegYZ : kColorSegXZ, 2));
                        item->setZValue(5);
                        items.push_back(item);
                    }
                }
            }
            state.items[key] = items;
        }
    }
}

std::vector<QPainterPath> OverlaySegmentationIntersections::buildPathsFromSegments(
    const std::vector<std::vector<cv::Vec3f>>& segments,
    PlaneSurface& plane,
    float coordScale,
    float breakDistance)
{
    std::vector<QPainterPath> paths;
    for (const auto& seg : segments) {
        if (seg.size() < 2) {
            continue;
        }

        QPainterPath path;
        bool first = true;
        cv::Vec3f last = {-1.f, -1.f, -1.f};
        for (const auto& wp : seg) {
            cv::Vec3f projected = plane.project(wp, 1.0f, coordScale);
            if (!std::isfinite(projected[0]) || !std::isfinite(projected[1])) {
                continue;
            }

            if (last[0] != -1.f) {
                const cv::Vec3f diff = projected - last;
                if (cv::norm(diff) >= breakDistance) {
                    if (!path.isEmpty()) {
                        paths.push_back(path);
                        path = QPainterPath();
                    }
                    first = true;
                }
            }

            if (first) {
                path.moveTo(projected[0], projected[1]);
            } else {
                path.lineTo(projected[0], projected[1]);
            }
            last = projected;
            first = false;
        }
        if (!path.isEmpty()) {
            paths.push_back(path);
        }
    }
    return paths;
}

OverlaySegmentationIntersections::ViewerState& OverlaySegmentationIntersections::ensureState(CVolumeViewer& viewer)
{
    auto [it, inserted] = _viewers.try_emplace(&viewer);
    if (inserted) {
        it->second.targets.insert("visible_segmentation");
    }
    return it->second;
}

const OverlaySegmentationIntersections::ViewerState*
OverlaySegmentationIntersections::lookupState(const CVolumeViewer& viewer) const
{
    auto it = _viewers.find(const_cast<CVolumeViewer*>(&viewer));
    if (it == _viewers.end()) {
        return nullptr;
    }
    return &it->second;
}

void OverlaySegmentationIntersections::clearItems(CVolumeViewer& viewer,
                                                   ViewerState& state,
                                                   const std::string& key)
{
    auto it = state.items.find(key);
    if (it == state.items.end()) {
        return;
    }
    for (auto* item : it->second) {
        if (!item) {
            continue;
        }
        if (auto* scene = viewer.scene()) {
            scene->removeItem(item);
        }
        delete item;
    }
    state.items.erase(it);
}

void OverlaySegmentationIntersections::clearAllItems(CVolumeViewer& viewer, ViewerState& state)
{
    std::vector<std::string> keys;
    keys.reserve(state.items.size());
    for (const auto& [key, _] : state.items) {
        keys.push_back(key);
    }
    for (const auto& key : keys) {
        clearItems(viewer, state, key);
    }
}
