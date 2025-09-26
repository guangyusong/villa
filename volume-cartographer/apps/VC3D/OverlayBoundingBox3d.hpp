#pragma once

#include <optional>
#include <vector>
#include <memory>
#include <filesystem>
#include <map>
#include <QString>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceVoxelizer.hpp"
#include "BBoxTypes.hpp"

class CWindow;
class CVolumeViewer;
class VolumePkg;
class Volume;
class ChunkCache;
class QWidget;
namespace z5 { class Dataset; }

class OverlayBoundingBox3d
{
public:
    OverlayBoundingBox3d(CWindow& window, ChunkCache* cache);
    ~OverlayBoundingBox3d() = default;

    void registerViewer(CVolumeViewer* viewer);
    void unregisterViewer(CVolumeViewer* viewer);

    void setVolumePackage(const std::shared_ptr<VolumePkg>& pkg);
    void clearVolume();

    void setEnabled(bool enabled);
    bool isEnabled() const { return _enabled; }

    std::optional<OrientedBBox> currentBox() const { return _sharedBox; }
    void clearBBox();

    void showCutOutDialog(QWidget* parent = nullptr);
    std::vector<std::string> segmentationsWithin(const OrientedBBox& bbox) const;
    std::vector<std::string> segmentationsInCurrentBBox() const;

private:
    void onViewerBBoxEdited(CVolumeViewer* source, const OrientedBBox& bbox, bool final);
    void refreshViewers(CVolumeViewer* source = nullptr);
    bool isSliceViewer(const std::string& surfName) const;

    std::string sanitizeForFilename(const std::string& name) const;
    bool exportToZarr(const std::string& volumeId, int level, const std::filesystem::path& outputDir, const OrientedBBox& bbox, QString& errorMessage);
    bool exportToTiff(const std::string& volumeId, int level, const std::filesystem::path& outputDir, const OrientedBBox& bbox, QString& errorMessage);
    bool exportSegmentationsToZarr(const std::vector<std::string>& segmentationIds,
                                   const std::string& volumeId,
                                   int level,
                                   const std::filesystem::path& outputDir,
                                   const OrientedBBox& bbox,
                                   QString& errorMessage) const;
    bool exportSegmentationsToTiff(const std::vector<std::string>& segmentationIds,
                                   const std::string& volumeId,
                                   int level,
                                   const std::filesystem::path& outputDir,
                                   const OrientedBBox& bbox,
                                   QString& errorMessage) const;
    bool buildShiftedSurfaces(const std::vector<std::string>& segmentationIds,
                              const OrientedBBox& bbox,
                              float levelScale,
                              std::map<std::string, QuadSurface*>& outSurfaces,
                              std::vector<std::unique_ptr<QuadSurface>>& storage,
                              QString& errorMessage) const;

    CWindow& _window;
    ChunkCache* _cache;
    std::weak_ptr<VolumePkg> _volumePkg;
    std::vector<CVolumeViewer*> _viewers;
    std::optional<OrientedBBox> _sharedBox;
    bool _enabled{false};
};
