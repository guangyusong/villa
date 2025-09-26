#include "OverlayHandler.hpp"

#include "OverlayBoundingBox3d.hpp"
#include "CWindow.hpp"
#include "CVolumeViewer.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Slicing.hpp"

OverlayHandler::OverlayHandler(CWindow& window, ChunkCache* cache)
    : _window(window)
    , _cache(cache)
    , _bbox(std::make_unique<OverlayBoundingBox3d>(window, cache))
    , _segmentations(std::make_unique<OverlaySegmentationIntersections>(*window.surfaceCollection()))
{
}

OverlayHandler::~OverlayHandler() = default;

void OverlayHandler::registerViewer(CVolumeViewer* viewer)
{
    if (!viewer || !_bbox) {
        return;
    }
    _bbox->registerViewer(viewer);
    if (_segmentations) {
        _segmentations->registerViewer(viewer);
    }
}

void OverlayHandler::setVolumePackage(const std::shared_ptr<VolumePkg>& pkg)
{
    if (_bbox) {
        _bbox->setVolumePackage(pkg);
    }
}

void OverlayHandler::clearVolume()
{
    if (_bbox) {
        _bbox->clearVolume();
    }
    if (_segmentations) {
        _segmentations->clear();
    }
}
