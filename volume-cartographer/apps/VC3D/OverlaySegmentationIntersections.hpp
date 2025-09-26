#pragma once

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <QPainterPath>

#include <opencv2/core.hpp>

class CVolumeViewer;
class PlaneSurface;
class QuadSurface;
struct Intersection;
class QGraphicsItem;
class CSurfaceCollection;

class OverlaySegmentationIntersections
{
public:
    explicit OverlaySegmentationIntersections(CSurfaceCollection& surfaces);
    ~OverlaySegmentationIntersections();

    void registerViewer(CVolumeViewer* viewer);
    void unregisterViewer(CVolumeViewer* viewer);

    void refreshViewer(CVolumeViewer& viewer);
    void invalidateViewer(CVolumeViewer& viewer, const std::optional<std::string>& key = std::nullopt);
    void hideViewerItems(CVolumeViewer& viewer);
    void clear();

    std::set<std::string> intersectsForViewer(const CVolumeViewer& viewer) const;
    void setIntersectsForViewer(CVolumeViewer& viewer, const std::set<std::string>& targets);

    void handleIntersectionChanged(CVolumeViewer& viewer,
                                   const std::string& a,
                                   const std::string& b,
                                   Intersection* intersection);

    static std::vector<QPainterPath> buildPathsFromSegments(const std::vector<std::vector<cv::Vec3f>>& segments,
                                                            PlaneSurface& plane,
                                                            float coordScale,
                                                            float breakDistance = 8.f);

private:
    struct ViewerState {
        std::set<std::string> targets;
        std::unordered_map<std::string, std::vector<QGraphicsItem*>> items;
        Intersection* ignoreIntersectionChange = nullptr;
    };

    ViewerState& ensureState(CVolumeViewer& viewer);
    const ViewerState* lookupState(const CVolumeViewer& viewer) const;
    void clearItems(CVolumeViewer& viewer, ViewerState& state, const std::string& key);
    void clearAllItems(CVolumeViewer& viewer, ViewerState& state);

    CSurfaceCollection& _surfaces;
    std::unordered_map<CVolumeViewer*, ViewerState> _viewers;
};
