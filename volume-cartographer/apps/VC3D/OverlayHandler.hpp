#pragma once

#include <memory>

class CWindow;
class CVolumeViewer;
class VolumePkg;
class ChunkCache;

#include "OverlayBoundingBox3d.hpp"
#include "OverlaySegmentationIntersections.hpp"

class OverlayHandler
{
public:
    OverlayHandler(CWindow& window, ChunkCache* cache);
    ~OverlayHandler();

    void registerViewer(CVolumeViewer* viewer);
    void setVolumePackage(const std::shared_ptr<VolumePkg>& pkg);
    void clearVolume();

    OverlayBoundingBox3d& bbox() { return *_bbox; }
    OverlaySegmentationIntersections& segmentationOverlay() { return *_segmentations; }

private:
    CWindow& _window;
    ChunkCache* _cache;
    std::unique_ptr<OverlayBoundingBox3d> _bbox;
    std::unique_ptr<OverlaySegmentationIntersections> _segmentations;
};
