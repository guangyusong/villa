#include "OverlayBoundingBox3d.hpp"

#include "CWindow.hpp"
#include "CVolumeViewer.hpp"
#include "OverlaySegmentationIntersections.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/xtensor_include.hpp"
#include <QObject>
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(containers, xtensor.hpp)
#include <xtensor/xadapt.hpp>

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QVariant>
#include <QDir>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QRadioButton>
#include <QStringList>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QImage>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <chrono>
#include <limits>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <z5/attributes.hxx>
#include <z5/dataset.hxx>
#include <z5/factory.hxx>
#include <z5/filesystem/handle.hxx>
#include <z5/multiarray/xtensor_access.hxx>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int STATUS_TIMEOUT_MS = 3000;
const std::array<const char*, 5> kSliceViewNames = {
    "seg xz", "seg yz", "xy plane", "xz plane", "yz plane"
};

enum class ExportFormat { Zarr, Tiff };

Rect3D normalizeRect(const Rect3D& rect)
{
    Rect3D out = rect;
    for (int axis = 0; axis < 3; ++axis) {
        if (out.low[axis] > out.high[axis]) {
            std::swap(out.low[axis], out.high[axis]);
        }
    }
    return out;
}

template <typename VecType>
std::string vecToString(const VecType& v)
{
    std::ostringstream oss;
    oss << '(' << v[0] << ',' << v[1] << ',' << v[2] << ')';
    return oss.str();
}

std::string shapeToString(const std::vector<std::size_t>& shape)
{
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) {
            oss << ',';
        }
    }
    oss << ']';
    return oss.str();
}


struct VolumeBlock {
    cv::Vec3i size{0, 0, 0};
    std::vector<uint8_t> data;
};

inline size_t linearIndex(const cv::Vec3i& size, int z, int y, int x)
{
    return static_cast<size_t>(z) * static_cast<size_t>(size[1]) * static_cast<size_t>(size[2])
         + static_cast<size_t>(y) * static_cast<size_t>(size[2])
         + static_cast<size_t>(x);
}

VolumeBlock makeBlockFromTensor(const xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& tensor,
                                const cv::Vec3i& size)
{
    VolumeBlock block;
    block.size = size;
    const size_t total = static_cast<size_t>(size[0]) * static_cast<size_t>(size[1]) * static_cast<size_t>(size[2]);
    block.data.resize(total, 0);
    size_t idx = 0;
    for (int z = 0; z < size[0]; ++z) {
        for (int y = 0; y < size[1]; ++y) {
            for (int x = 0; x < size[2]; ++x) {
                block.data[idx++] = tensor(z, y, x);
            }
        }
    }
    return block;
}

VolumeBlock downsampleBlock(const VolumeBlock& src)
{
    if (src.size[0] <= 1 && src.size[1] <= 1 && src.size[2] <= 1) {
        return src;
    }

    VolumeBlock dst;
    dst.size[0] = std::max(1, (src.size[0] + 1) / 2);
    dst.size[1] = std::max(1, (src.size[1] + 1) / 2);
    dst.size[2] = std::max(1, (src.size[2] + 1) / 2);
    const size_t total = static_cast<size_t>(dst.size[0]) * static_cast<size_t>(dst.size[1]) * static_cast<size_t>(dst.size[2]);
    dst.data.resize(total, 0);

    for (int z = 0; z < dst.size[0]; ++z) {
        for (int y = 0; y < dst.size[1]; ++y) {
            for (int x = 0; x < dst.size[2]; ++x) {
                const int baseZ = z * 2;
                const int baseY = y * 2;
                const int baseX = x * 2;
                int sum = 0;
                int count = 0;
                for (int dz = 0; dz < 2 && (baseZ + dz) < src.size[0]; ++dz) {
                    for (int dy = 0; dy < 2 && (baseY + dy) < src.size[1]; ++dy) {
                        for (int dx = 0; dx < 2 && (baseX + dx) < src.size[2]; ++dx) {
                            size_t srcIdx = linearIndex(src.size, baseZ + dz, baseY + dy, baseX + dx);
                            sum += src.data[srcIdx];
                            ++count;
                        }
                    }
                }
                size_t dstIdx = linearIndex(dst.size, z, y, x);
                dst.data[dstIdx] = static_cast<uint8_t>((sum + count / 2) / std::max(1, count));
            }
        }
    }

    return dst;
}

xt::xarray<uint8_t> blockToArray(const VolumeBlock& block)
{
    const size_t Z = static_cast<size_t>(block.size[0]);
    const size_t Y = static_cast<size_t>(block.size[1]);
    const size_t X = static_cast<size_t>(block.size[2]);
    xt::xarray<uint8_t> arr = xt::zeros<uint8_t>({Z, Y, X});
    size_t idx = 0;
    for (size_t z = 0; z < Z; ++z) {
        for (size_t y = 0; y < Y; ++y) {
            for (size_t x = 0; x < X; ++x) {
                arr(z, y, x) = block.data[idx++];
            }
        }
    }
    return arr;
}

} // namespace

namespace {

struct SamplingGrid {
    int width{1};
    int height{1};
    int depth{1};
    float stepU{0.f};
    float stepV{0.f};
    float stepN{0.f};
};

inline SamplingGrid makeSamplingGrid(const OrientedBBox& box)
{
    SamplingGrid grid;
    const float minSpan = 1.f;

    auto computeExtent = [&](float halfExtent) {
        if (halfExtent < 1e-4f) {
            return 1;
        }
        float span = std::max(minSpan, std::max(halfExtent * 2.f, minSpan));
        int voxels = std::max(1, static_cast<int>(std::ceil(span)));
        // Include both bounding faces by adding one more sample along the axis
        return voxels + 1;
    };

    grid.width = computeExtent(box.halfExtents[0]);
    grid.height = computeExtent(box.halfExtents[1]);
    grid.depth = (box.halfExtents[2] < 1e-4f)
        ? 1
        : std::max(1, static_cast<int>(std::ceil(std::max(box.halfExtents[2] * 2.f, 1.f)))) + 1;

    grid.stepU = (grid.width > 1 && box.halfExtents[0] > 1e-4f)
        ? (2.f * box.halfExtents[0]) / static_cast<float>(grid.width - 1)
        : 0.f;
    grid.stepV = (grid.height > 1 && box.halfExtents[1] > 1e-4f)
        ? (2.f * box.halfExtents[1]) / static_cast<float>(grid.height - 1)
        : 0.f;
    grid.stepN = (grid.depth > 1 && box.halfExtents[2] > 1e-4f)
        ? (2.f * box.halfExtents[2]) / static_cast<float>(grid.depth - 1)
        : 0.f;

    return grid;
}

inline float coordinateAt(int index, int count, float halfExtent, float step)
{
    if (count <= 1 || halfExtent < 1e-4f || step < 1e-6f) {
        return 0.f;
    }
    float start = -halfExtent;
    return start + static_cast<float>(index) * step;
}

inline float gridIndexForValue(float value, int count, float halfExtent, float step)
{
    if (count <= 1 || halfExtent < 1e-4f || step < 1e-6f) {
        return (count - 1) * 0.5f;
    }
    float start = -halfExtent;
    return (value - start) / step;
}

inline cv::Vec3f toDatasetCoords(const cv::Vec3f& world,
                                 const std::array<double, 3>& scaleFactors,
                                 const std::array<double, 3>& translations)
{
    cv::Vec3f out;
    for (int axis = 0; axis < 3; ++axis) {
        const double scale = scaleFactors[axis];
        const double translation = translations[axis];
        if (std::abs(scale) < 1e-9) {
            out[axis] = static_cast<float>(world[axis] - translation);
        } else {
            out[axis] = static_cast<float>((world[axis] - translation) / scale);
        }
    }
    return out;
}

inline cv::Vec3f normalizeVec(const cv::Vec3f& v)
{
    float norm = cv::norm(v);
    if (norm < 1e-6f) {
        return v;
    }
    return v / norm;
}

constexpr float kLoopClosureTolerance = 1.5f;

struct SurfaceProjection {
    QuadSurface* surface = nullptr;
    const cv::Mat_<cv::Vec3f>* points = nullptr;
    cv::Rect roi;
    int zMin = 0;
    int zMax = -1;
    std::string id;
};

inline std::vector<std::size_t> samplingGridShape(const SamplingGrid& grid)
{
    return {
        static_cast<std::size_t>(std::max(1, grid.depth)),
        static_cast<std::size_t>(std::max(1, grid.height)),
        static_cast<std::size_t>(std::max(1, grid.width))
    };
}

inline cv::Point clampPointToSlice(cv::Point pt, int width, int height)
{
    if (width <= 0 || height <= 0) {
        return {0, 0};
    }
    pt.x = std::clamp(pt.x, 0, width - 1);
    pt.y = std::clamp(pt.y, 0, height - 1);
    return pt;
}

void drawSegmentOnSlice(cv::Mat& slice,
                        const std::vector<cv::Vec3f>& segment,
                        PlaneSurface& plane,
                        int width,
                        int height)
{
    if (segment.size() < 2 || width <= 0 || height <= 0) {
        return;
    }

    std::vector<cv::Point> contour;
    contour.reserve(segment.size());

    for (const auto& point3d : segment) {
        if (!std::isfinite(point3d[0]) || !std::isfinite(point3d[1]) || !std::isfinite(point3d[2])) {
            continue;
        }
        cv::Vec3f projected = plane.project(point3d, 1.0f, 1.0f);
        if (!std::isfinite(projected[0]) || !std::isfinite(projected[1])) {
            continue;
        }
        cv::Point pixel{static_cast<int>(std::lround(projected[0])), static_cast<int>(std::lround(projected[1]))};
        contour.push_back(clampPointToSlice(pixel, width, height));
    }

    if (contour.empty()) {
        return;
    }

    if (contour.size() == 1) {
        const cv::Point& p = contour.front();
        slice.at<uint8_t>(p.y, p.x) = 255;
        return;
    }

    bool isClosed = false;
    if (segment.size() >= 3) {
        const cv::Vec3f diff = segment.front() - segment.back();
        isClosed = cv::norm(diff) <= kLoopClosureTolerance;
    }

    std::vector<std::vector<cv::Point>> polyline{contour};
    const int thickness = 3;
    cv::polylines(slice, polyline, isClosed, cv::Scalar(255), thickness, cv::LINE_AA);
}

bool voxelizeSegmentationsViewerStyle(const std::map<std::string, QuadSurface*>& surfaceMap,
                                      const SamplingGrid& grid,
                                      std::vector<uint8_t>& volumeData)
{
    if (grid.width <= 0 || grid.height <= 0 || grid.depth <= 0) {
        volumeData.clear();
        return false;
    }

    const int width = std::max(1, grid.width);
    const int height = std::max(1, grid.height);
    const int depth = std::max(1, grid.depth);
    const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(depth);

    volumeData.assign(total, 0);

    if (surfaceMap.empty()) {
        return true;
    }

    const cv::Rect fullROI(0, 0, width, height);

    std::vector<SurfaceProjection> projections;
    projections.reserve(surfaceMap.size());

    for (const auto& [segId, surface] : surfaceMap) {
        if (!surface) {
            continue;
        }

        const cv::Mat_<cv::Vec3f>* pointsPtr = surface->rawPointsPtr();
        if (!pointsPtr) {
            continue;
        }
        const auto& points = *pointsPtr;

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();

        for (int r = 0; r < points.rows; ++r) {
            for (int c = 0; c < points.cols; ++c) {
                const cv::Vec3f& p = points(r, c);
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || p[0] < 0.f) {
                    continue;
                }
                minX = std::min(minX, p[0]);
                minY = std::min(minY, p[1]);
                minZ = std::min(minZ, p[2]);
                maxX = std::max(maxX, p[0]);
                maxY = std::max(maxY, p[1]);
                maxZ = std::max(maxZ, p[2]);
            }
        }

        if (!std::isfinite(minX) || !std::isfinite(minY) || !std::isfinite(minZ)) {
            continue;
        }

        const int zMin = std::max(0, static_cast<int>(std::floor(minZ)) - 1);
        const int zMax = std::min(depth - 1, static_cast<int>(std::ceil(maxZ)) + 1);
        if (zMin > zMax) {
            continue;
        }

        cv::Rect surfaceROI(
            std::max(0, static_cast<int>(std::floor(minX)) - 1),
            std::max(0, static_cast<int>(std::floor(minY)) - 1),
            std::max(1, static_cast<int>(std::ceil(maxX)) - std::max(0, static_cast<int>(std::floor(minX)) - 1) + 2),
            std::max(1, static_cast<int>(std::ceil(maxY)) - std::max(0, static_cast<int>(std::floor(minY)) - 1) + 2)
        );
        surfaceROI &= fullROI;
        if (surfaceROI.width <= 0 || surfaceROI.height <= 0) {
            continue;
        }

        projections.push_back({surface, pointsPtr, surfaceROI, zMin, zMax, segId});
    }

    if (projections.empty()) {
        return true;
    }

    std::vector<std::vector<int>> surfacesBySlice(static_cast<std::size_t>(depth));
    for (int idx = 0; idx < static_cast<int>(projections.size()); ++idx) {
        const auto& proj = projections[static_cast<std::size_t>(idx)];
        const int zStart = std::max(0, proj.zMin);
        const int zEnd = std::min(depth - 1, proj.zMax);
        for (int z = zStart; z <= zEnd; ++z) {
            surfacesBySlice[static_cast<std::size_t>(z)].push_back(idx);
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < depth; ++z) {
        const auto& surfaceIndices = surfacesBySlice[static_cast<std::size_t>(z)];
        if (surfaceIndices.empty()) {
            continue;
        }

        QImage sliceMask(width, height, QImage::Format_Grayscale8);
        sliceMask.fill(Qt::black);
        QPainter painter(&sliceMask);
        painter.setRenderHint(QPainter::Antialiasing, true);

        PlaneSurface slicePlane({0.f, 0.f, static_cast<float>(z)}, {0.f, 0.f, 1.f});
        const std::size_t base = static_cast<std::size_t>(z) * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

        for (int idx : surfaceIndices) {
            const auto& proj = projections[static_cast<std::size_t>(idx)];
            if (!proj.surface || !proj.points) {
                continue;
            }

            std::vector<std::vector<cv::Vec3f>> intersections;
            std::vector<std::vector<cv::Vec2f>> unusedGrid;
            const bool isPrimarySeg = (proj.id == "segmentation");
            const float step = 2.0f;
            const int minTries = 1000;

            find_intersect_segments(intersections,
                                    unusedGrid,
                                    *(proj.points),
                                    &slicePlane,
                                    proj.roi,
                                    step,
                                    minTries);

            if (intersections.empty()) {
                continue;
            }

            painter.save();
            painter.setClipRect(QRect(proj.roi.x, proj.roi.y, proj.roi.width, proj.roi.height));

            const auto paths = OverlaySegmentationIntersections::buildPathsFromSegments(intersections, slicePlane, 1.0f);
            if (!paths.empty()) {
                QPen pen(Qt::white, isPrimarySeg ? 3.0 : 2.0);
                painter.setPen(pen);
                painter.setBrush(Qt::NoBrush);
                for (const auto& path : paths) {
                    painter.drawPath(path);
                }
            }
            painter.restore();
        }

        painter.end();

        for (int y = 0; y < height; ++y) {
            const uint8_t* srcRow = sliceMask.constScanLine(y);
            uint8_t* dstRow = volumeData.data() + base + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                if (srcRow[x]) {
                    dstRow[x] = 255;
                }
            }
        }
    }

    return true;
}

void downsampleViewerStyleLevel(z5::filesystem::handle::File& zarrFile, int targetLevel)
{
    auto srcDs = z5::openDataset(zarrFile, std::to_string(targetLevel - 1));
    auto dstDs = z5::openDataset(zarrFile, std::to_string(targetLevel));

    const auto& srcShape = srcDs->shape();
    const auto& dstShape = dstDs->shape();

    constexpr std::size_t chunkSize = 64;

    for (std::size_t dz = 0; dz < dstShape[0]; dz += chunkSize) {
        for (std::size_t dy = 0; dy < dstShape[1]; dy += chunkSize) {
            for (std::size_t dx = 0; dx < dstShape[2]; dx += chunkSize) {
                const std::size_t chunkNz = std::min(chunkSize, dstShape[0] - dz);
                const std::size_t chunkNy = std::min(chunkSize, dstShape[1] - dy);
                const std::size_t chunkNx = std::min(chunkSize, dstShape[2] - dx);

                xt::xarray<uint8_t> dstChunk = xt::zeros<uint8_t>({chunkNz, chunkNy, chunkNx});

                const std::size_t srcZ = dz * 2;
                const std::size_t srcY = dy * 2;
                const std::size_t srcX = dx * 2;
                const std::size_t srcNz = std::min(chunkNz * 2, srcShape[0] - srcZ);
                const std::size_t srcNy = std::min(chunkNy * 2, srcShape[1] - srcY);
                const std::size_t srcNx = std::min(chunkNx * 2, srcShape[2] - srcX);

                xt::xarray<uint8_t> srcChunk = xt::zeros<uint8_t>({srcNz, srcNy, srcNx});
                z5::types::ShapeType srcOffset = {srcZ, srcY, srcX};
                z5::multiarray::readSubarray<uint8_t>(srcDs, srcChunk, srcOffset.begin());

                for (std::size_t z = 0; z < chunkNz; ++z) {
                    for (std::size_t y = 0; y < chunkNy; ++y) {
                        for (std::size_t x = 0; x < chunkNx; ++x) {
                            uint8_t maxVal = 0;
                            for (int dz2 = 0; dz2 < 2 && z * 2 + dz2 < srcNz; ++dz2) {
                                for (int dy2 = 0; dy2 < 2 && y * 2 + dy2 < srcNy; ++dy2) {
                                    for (int dx2 = 0; dx2 < 2 && x * 2 + dx2 < srcNx; ++dx2) {
                                        maxVal = std::max(maxVal, srcChunk(z * 2 + dz2, y * 2 + dy2, x * 2 + dx2));
                                    }
                                }
                            }
                            dstChunk(z, y, x) = maxVal;
                        }
                    }
                }

                z5::types::ShapeType dstOffset = {dz, dy, dx};
                z5::multiarray::writeSubarray<uint8_t>(dstDs, dstChunk, dstOffset.begin());
            }
        }
    }
}

void createViewerStylePyramid(z5::filesystem::handle::File& zarrFile,
                              const std::vector<std::size_t>& baseShape)
{
    for (int level = 1; level < 5; ++level) {
        const int scale = 1 << level;
        std::vector<std::size_t> shape = {
            (baseShape[0] + scale - 1) / scale,
            (baseShape[1] + scale - 1) / scale,
            (baseShape[2] + scale - 1) / scale
        };

        std::vector<std::size_t> chunks = {64, 64, 64};
        for (std::size_t i = 0; i < 3; ++i) {
            chunks[i] = std::max<std::size_t>(1, std::min<std::size_t>(chunks[i], shape[i]));
        }

        z5::createDataset(zarrFile, std::to_string(level), "uint8", shape, chunks);
        downsampleViewerStyleLevel(zarrFile, level);
    }
}

bool writeViewerStyleVolumeToZarr(const SamplingGrid& grid,
                                  const std::vector<uint8_t>& volumeData,
                                  const std::filesystem::path& outPath,
                                  QString& errorMessage)
{
    try {
        std::filesystem::create_directories(outPath);
    } catch (const std::exception& e) {
        errorMessage = QObject::tr("Failed to create output directory: %1").arg(e.what());
        return false;
    }

    z5::filesystem::handle::File outFile(outPath);
    z5::createFile(outFile, true);

    const auto shape = samplingGridShape(grid);
    std::vector<std::size_t> chunks(3);
    for (std::size_t i = 0; i < 3; ++i) {
        chunks[i] = std::max<std::size_t>(1, std::min<std::size_t>(64, shape[i]));
    }

    auto ds0 = z5::createDataset(outFile, "0", "uint8", shape, chunks);

    auto adapted = xt::adapt(volumeData, shape);
    z5::types::ShapeType offset = {0, 0, 0};
    z5::multiarray::writeSubarray<uint8_t>(ds0, adapted, offset.begin());

    createViewerStylePyramid(outFile, shape);

    return true;
}

inline std::optional<VolumeBlock> sampleIntensityBlock(z5::Dataset* dataset,
                                                       ChunkCache* cache,
                                                       const OrientedBBox& box,
                                                       const std::array<double, 3>& scaleFactors,
                                                       const std::array<double, 3>& translations,
                                                       QString& errorMessage)
{
    if (!dataset) {
        errorMessage = QObject::tr("Dataset unavailable for selected level");
        return std::nullopt;
    }
    if (!cache) {
        errorMessage = QObject::tr("Chunk cache unavailable");
        return std::nullopt;
    }

    VolumeBlock block;
    SamplingGrid grid = makeSamplingGrid(box);
    block.size = {grid.depth, grid.height, grid.width};
    const size_t total = static_cast<size_t>(grid.depth) * grid.height * grid.width;
    block.data.assign(total, 0);

    const cv::Vec3f axisU = normalizeVec(box.axisU);
    const cv::Vec3f axisV = normalizeVec(box.axisV);
    const cv::Vec3f axisN = normalizeVec(box.axisN);

    cv::Mat_<uint8_t> slice(grid.height, grid.width, uint8_t(0));
    cv::Mat_<cv::Vec3f> coords(grid.height, grid.width);

    cv::Vec3f minWorld(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Vec3f maxWorld(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    cv::Vec3f minDsCoord(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Vec3f maxDsCoord(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    for (int z = 0; z < grid.depth; ++z) {
        float offsetN = coordinateAt(z, grid.depth, box.halfExtents[2], grid.stepN);

        for (int y = 0; y < grid.height; ++y) {
            float offsetV = coordinateAt(y, grid.height, box.halfExtents[1], grid.stepV);
            for (int x = 0; x < grid.width; ++x) {
                float offsetU = coordinateAt(x, grid.width, box.halfExtents[0], grid.stepU);
                cv::Vec3f world = box.center
                    + axisU * offsetU
                    + axisV * offsetV
                    + axisN * offsetN;
                cv::Vec3f dsCoord = toDatasetCoords(world, scaleFactors, translations);
                coords(y, x) = cv::Vec3f(dsCoord[0], dsCoord[1], dsCoord[2]);

                for (int axis = 0; axis < 3; ++axis) {
                    minWorld[axis] = std::min(minWorld[axis], world[axis]);
                    maxWorld[axis] = std::max(maxWorld[axis], world[axis]);
                    minDsCoord[axis] = std::min(minDsCoord[axis], dsCoord[axis]);
                    maxDsCoord[axis] = std::max(maxDsCoord[axis], dsCoord[axis]);
                }
            }
        }

        if (z == 0) {
            const auto& chunkShape = dataset->chunking().blockShape();
            const auto& shape = dataset->shape();
            const cv::Vec3f sample = coords(0, 0);
            int cz = static_cast<int>(chunkShape[0]);
            int cy = static_cast<int>(chunkShape[1]);
            int cx = static_cast<int>(chunkShape[2]);
            int chunkZ = static_cast<int>(sample[2]) / std::max(1, cz);
            int chunkY = static_cast<int>(sample[1]) / std::max(1, cy);
            int chunkX = static_cast<int>(sample[0]) / std::max(1, cx);
            z5::types::ShapeType chunkId = {
                static_cast<std::size_t>(chunkZ),
                static_cast<std::size_t>(chunkY),
                static_cast<std::size_t>(chunkX)
            };
            bool exists = dataset->chunkExists(chunkId);
            std::cout << "[Cutout] coord sample (X,Y,Z)=" << '(' << sample[0] << ',' << sample[1] << ',' << sample[2] << ')'
                      << " chunkShape(Z,Y,X)=" << '(' << chunkShape[0] << ',' << chunkShape[1] << ',' << chunkShape[2] << ')'
                      << " datasetShape=" << shapeToString(shape)
                      << " chunkId(Z,Y,X)=" << '(' << chunkZ << ',' << chunkY << ',' << chunkX << ')'
                      << " chunkExists=" << std::boolalpha << exists
                      << std::noboolalpha << std::endl;
        }

        readInterpolated3D(slice, dataset, coords, cache);

        size_t base = static_cast<size_t>(z) * grid.height * grid.width;
        for (int y = 0; y < grid.height; ++y) {
            const uint8_t* row = slice.ptr<uint8_t>(y);
            std::copy(row, row + grid.width, block.data.begin() + base + static_cast<size_t>(y) * grid.width);
        }
    }

    std::cout << "[Cutout] sampleIntensityBlock grid width=" << grid.width
              << " height=" << grid.height
              << " depth=" << grid.depth
              << " stepU=" << grid.stepU
              << " stepV=" << grid.stepV
              << " stepN=" << grid.stepN
              << " worldMin=" << vecToString(minWorld)
              << " worldMax=" << vecToString(maxWorld)
              << " dsCoordMin=" << vecToString(minDsCoord)
              << " dsCoordMax=" << vecToString(maxDsCoord)
              << std::endl;

    return block;
}

struct DatasetRange
{
    std::array<double, 3> min{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()};
    std::array<double, 3> max{
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()};
};

DatasetRange datasetRangeForBox(const OrientedBBox& box,
                                const std::array<double, 3>& scaleFactors,
                                const std::array<double, 3>& translations)
{
    DatasetRange range;
    const cv::Vec3f axisU = normalizeVec(box.axisU);
    const cv::Vec3f axisV = normalizeVec(box.axisV);
    const cv::Vec3f axisN = normalizeVec(box.axisN);

    for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
            for (int sz : {-1, 1}) {
                cv::Vec3f corner = box.center
                    + axisU * (static_cast<float>(sx) * box.halfExtents[0])
                    + axisV * (static_cast<float>(sy) * box.halfExtents[1])
                    + axisN * (static_cast<float>(sz) * box.halfExtents[2]);
                cv::Vec3f ds = toDatasetCoords(corner, scaleFactors, translations);
                for (int axis = 0; axis < 3; ++axis) {
                    range.min[axis] = std::min(range.min[axis], static_cast<double>(ds[axis]));
                    range.max[axis] = std::max(range.max[axis], static_cast<double>(ds[axis]));
                }
            }
        }
    }

    return range;
}

std::string coordinateFileStem(z5::Dataset* dataset,
                               const OrientedBBox& box,
                               const std::array<double, 3>& scaleFactors,
                               const std::array<double, 3>& translations,
                               int level,
                               const std::string& suffix)
{
    if (!dataset) {
        std::ostringstream oss;
        oss << "bbox";
        if (level != 0) {
            oss << "_L" << level;
        }
        if (!suffix.empty()) {
            oss << suffix;
        }
        return oss.str();
    }

    DatasetRange range = datasetRangeForBox(box, scaleFactors, translations);
    const auto& shape = dataset->shape();
    if (shape.size() < 3) {
        std::ostringstream fallback;
        fallback << "bbox";
        if (level != 0) {
            fallback << "_L" << level;
        }
        if (!suffix.empty()) {
            fallback << suffix;
        }
        return fallback.str();
    }

    std::array<int, 3> minIdx{};
    std::array<int, 3> maxIdx{};
    for (int axis = 0; axis < 3; ++axis) {
        double minVal = range.min[axis];
        double maxVal = range.max[axis];
        if (minVal > maxVal) {
            std::swap(minVal, maxVal);
        }
        int axisMax = static_cast<int>(std::max<std::size_t>(1, shape[axis])) - 1;
        int minCandidate = static_cast<int>(std::floor(minVal));
        int maxCandidate = static_cast<int>(std::ceil(maxVal));
        minCandidate = std::clamp(minCandidate, 0, axisMax);
        maxCandidate = std::clamp(maxCandidate, minCandidate, axisMax);
        minIdx[axis] = minCandidate;
        maxIdx[axis] = maxCandidate;
    }

    std::ostringstream oss;
    oss << 'z' << minIdx[0] << '-' << maxIdx[0]
        << "_y" << minIdx[1] << '-' << maxIdx[1]
        << "_x" << minIdx[2] << '-' << maxIdx[2];
    if (level != 0) {
        oss << "_L" << level;
    }
    if (!suffix.empty()) {
        oss << suffix;
    }
    return oss.str();
}

} // namespace

OverlayBoundingBox3d::OverlayBoundingBox3d(CWindow& window, ChunkCache* cache)
    : _window(window)
    , _cache(cache)
{
}

void OverlayBoundingBox3d::registerViewer(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }
    if (std::find(_viewers.begin(), _viewers.end(), viewer) != _viewers.end()) {
        return;
    }

    viewer->setBBoxCallbacks(
        [this, viewer](const OrientedBBox& bbox, bool final) {
            onViewerBBoxEdited(viewer, bbox, final);
        },
        [this]() -> std::optional<OrientedBBox> {
            return _sharedBox;
        });

    _viewers.push_back(viewer);
    refreshViewers(viewer);
}

void OverlayBoundingBox3d::unregisterViewer(CVolumeViewer* viewer)
{
    const auto it = std::remove(_viewers.begin(), _viewers.end(), viewer);
    if (it != _viewers.end()) {
        _viewers.erase(it, _viewers.end());
    }
}

void OverlayBoundingBox3d::setVolumePackage(const std::shared_ptr<VolumePkg>& pkg)
{
    _volumePkg = pkg;
}

void OverlayBoundingBox3d::clearVolume()
{
    _sharedBox.reset();
    refreshViewers();
    _volumePkg.reset();
}

void OverlayBoundingBox3d::clearBBox()
{
    if (_sharedBox.has_value()) {
        _sharedBox.reset();
    }
    refreshViewers();
}

void OverlayBoundingBox3d::setEnabled(bool enabled)
{
    if (_enabled == enabled) {
        return;
    }
    _enabled = enabled;
    refreshViewers();

    if (auto* bar = _window.statusBar()) {
        bar->showMessage(
            enabled ? QObject::tr("BBox mode active: adjust the 3D box in slice views")
                     : QObject::tr("BBox mode off"),
            STATUS_TIMEOUT_MS);
    }
}

void OverlayBoundingBox3d::onViewerBBoxEdited(CVolumeViewer* source, const OrientedBBox& bbox, bool /*final*/)
{
    _sharedBox = bbox;
    refreshViewers(source);
}

void OverlayBoundingBox3d::refreshViewers(CVolumeViewer* source)
{
    for (auto* viewer : _viewers) {
        if (!viewer) {
            continue;
        }
        viewer->setBBoxMode(_enabled);
        const bool isSlice = isSliceViewer(viewer->surfName());
        if (!_enabled || !isSlice) {
            viewer->setExternalBBox(std::nullopt);
        } else if (viewer != source) {
            viewer->setExternalBBox(_sharedBox);
        }
    }
}

bool OverlayBoundingBox3d::isSliceViewer(const std::string& surfName) const
{
    return std::any_of(kSliceViewNames.begin(), kSliceViewNames.end(),
        [&surfName](const char* name) {
            return surfName == name;
        });
}

std::string OverlayBoundingBox3d::sanitizeForFilename(const std::string& name) const
{
    std::string result;
    result.reserve(name.size());
    for (unsigned char ch : name) {
        if (std::isalnum(ch) || ch == '_' || ch == '-') {
            result.push_back(static_cast<char>(ch));
        } else if (std::isspace(ch)) {
            result.push_back('_');
        }
    }
    if (result.empty()) {
        result = "volume";
    }
    return result;
}

bool OverlayBoundingBox3d::exportToZarr(const std::string& volumeId,
                                        int level,
                                        const std::filesystem::path& outputDir,
                                        const OrientedBBox& bbox,
                                        QString& errorMessage)
{
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        errorMessage = QObject::tr("No volume package loaded");
        return false;
    }
    if (!_cache) {
        errorMessage = QObject::tr("Chunk cache unavailable");
        return false;
    }

    auto volumePtr = volumePkg->volume(volumeId);
    if (!volumePtr) {
        errorMessage = QObject::tr("Volume %1 is not available").arg(QString::fromStdString(volumeId));
        return false;
    }

    z5::Dataset* dataset = volumePtr->zarrDataset(level);
    QString samplingError;
    const auto scaleFactors = volumePtr->levelScaleFactors(level);
    const auto translations = volumePtr->levelTranslations(level);
    if (dataset) {
        std::cout << "[Cutout] exportToZarr volume=" << volumeId
                  << " level=" << level
                  << " datasetShape=" << shapeToString(dataset->shape())
                  << " bboxCenter=" << vecToString(bbox.center)
                  << " bboxHalfExtents=" << vecToString(bbox.halfExtents)
                  << " axisU=" << vecToString(bbox.axisU)
                  << " axisV=" << vecToString(bbox.axisV)
                  << " axisN=" << vecToString(bbox.axisN)
                  << " scaleFactors=" << '(' << scaleFactors[0] << ',' << scaleFactors[1] << ',' << scaleFactors[2] << ')'
                  << " translations=" << '(' << translations[0] << ',' << translations[1] << ',' << translations[2] << ')'
                  << std::endl;
    } else {
        std::cout << "[Cutout] exportToZarr volume=" << volumeId
                  << " level=" << level
                  << " dataset=null" << std::endl;
    }
    auto blockOpt = sampleIntensityBlock(dataset, _cache, bbox, scaleFactors, translations, samplingError);
    if (!blockOpt) {
        errorMessage = samplingError;
        return false;
    }
    VolumeBlock block = std::move(*blockOpt);

    auto minmax = std::minmax_element(block.data.begin(), block.data.end());
    const auto nonZero = static_cast<size_t>(std::count_if(block.data.begin(), block.data.end(), [](uint8_t v) { return v != 0; }));
    std::cout << "[Cutout] sampled block size=" << vecToString(block.size)
              << " totalVoxels=" << block.data.size()
              << " min=" << static_cast<int>(*minmax.first)
              << " max=" << static_cast<int>(*minmax.second)
              << " nonZero=" << nonZero
              << std::endl;

    std::vector<VolumeBlock> pyramid;
    pyramid.reserve(6);
    pyramid.emplace_back(block);
    while (pyramid.size() < 6) {
        VolumeBlock next = downsampleBlock(pyramid.back());
        pyramid.push_back(next);
        if (pyramid.back().size == pyramid[pyramid.size() - 2].size) {
            while (pyramid.size() < 6) {
                pyramid.push_back(pyramid.back());
            }
            break;
        }
    }

    std::string fileStem = coordinateFileStem(dataset, bbox, scaleFactors, translations, level, "");
    if (fileStem.empty()) {
        fileStem = "bbox";
        if (level != 0) {
            fileStem += "_L" + std::to_string(level);
        }
    }
    std::filesystem::path outPath = outputDir / (fileStem + ".zarr");

    if (std::filesystem::exists(outPath)) {
        errorMessage = QObject::tr("Output path %1 already exists")
            .arg(QString::fromStdString(outPath.string()));
        return false;
    }

    try {
        std::filesystem::create_directories(outPath);
    } catch (const std::exception& e) {
        errorMessage = QObject::tr("Failed to create output directory: %1").arg(e.what());
        return false;
    }

    z5::filesystem::handle::File outFile(outPath);
    z5::createFile(outFile, true);

    nlohmann::json compOpts = {
        {"cname", "zstd"},
        {"clevel", 1},
        {"shuffle", 0}
    };

    for (size_t lvl = 0; lvl < pyramid.size(); ++lvl) {
        const auto& blk = pyramid[lvl];
        std::vector<size_t> shape = {
            static_cast<size_t>(blk.size[0]),
            static_cast<size_t>(blk.size[1]),
            static_cast<size_t>(blk.size[2])
        };
        std::vector<size_t> chunks = {
            std::max<size_t>(1, std::min<size_t>(64, shape[0])),
            std::max<size_t>(1, std::min<size_t>(128, shape[1])),
            std::max<size_t>(1, std::min<size_t>(128, shape[2]))
        };

        auto dsOut = z5::createDataset(outFile, std::to_string(lvl), "uint8", shape, chunks, std::string("blosc"), compOpts);
        xt::xarray<uint8_t> arr = xt::zeros<uint8_t>({shape[0], shape[1], shape[2]});
        size_t idx = 0;
        for (size_t z = 0; z < shape[0]; ++z) {
            for (size_t y = 0; y < shape[1]; ++y) {
                for (size_t x = 0; x < shape[2]; ++x) {
                    arr(z, y, x) = blk.data[idx++];
                }
            }
        }
        z5::types::ShapeType offset = {0, 0, 0};
        z5::multiarray::writeSubarray<uint8_t>(dsOut, arr, offset.begin());
    }

    double voxSize = volumePtr->voxelSize();
    const float effectiveVoxelSize = voxSize > 0.0 ? static_cast<float>(voxSize * scaleFactors[0]) : 1.0f;

    SamplingGrid grid = makeSamplingGrid(bbox);

    nlohmann::json orientation;
    orientation["center"] = {bbox.center[0], bbox.center[1], bbox.center[2]};
    orientation["axis_u"] = {bbox.axisU[0], bbox.axisU[1], bbox.axisU[2]};
    orientation["axis_v"] = {bbox.axisV[0], bbox.axisV[1], bbox.axisV[2]};
    orientation["axis_n"] = {bbox.axisN[0], bbox.axisN[1], bbox.axisN[2]};
    orientation["half_extents"] = {bbox.halfExtents[0], bbox.halfExtents[1], bbox.halfExtents[2]};
    orientation["grid_shape"] = {block.size[2], block.size[1], block.size[0]};
    orientation["grid_step"] = {grid.stepU, grid.stepV, grid.stepN};
    orientation["level"] = level;

    nlohmann::json attrs;
    attrs["source_volume_id"] = volumeId;
    attrs["voxel_size_level"] = effectiveVoxelSize;
    attrs["orientation"] = orientation;

    nlohmann::json multiscale;
    multiscale["version"] = "0.4";
    multiscale["name"] = "bbox";
    multiscale["axes"] = nlohmann::json::array({
        nlohmann::json{{"name","z"},{"type","space"},{"unit","pixel"}},
        nlohmann::json{{"name","y"},{"type","space"},{"unit","pixel"}},
        nlohmann::json{{"name","x"},{"type","space"},{"unit","pixel"}}
    });

    nlohmann::json datasets = nlohmann::json::array();
    for (size_t lvl = 0; lvl < pyramid.size(); ++lvl) {
        double scale = std::pow(2.0, static_cast<double>(lvl));
        nlohmann::json dset;
        dset["path"] = std::to_string(lvl);
        dset["coordinateTransformations"] = nlohmann::json::array({
            nlohmann::json{{"type","scale"},{"scale", nlohmann::json::array({scale, scale, scale})}},
            nlohmann::json{{"type","translation"},{"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}
        });
        datasets.push_back(dset);
    }
    multiscale["datasets"] = datasets;
    multiscale["metadata"] = nlohmann::json{{"downsampling_method", "mean"}};

    attrs["multiscales"] = nlohmann::json::array({multiscale});

    try {
        z5::filesystem::writeAttributes(outFile, attrs);
    } catch (const std::exception& e) {
        errorMessage = QObject::tr("Failed to write zarr attributes: %1").arg(e.what());
        return false;
    }

    return true;
}


bool OverlayBoundingBox3d::exportToTiff(const std::string& volumeId,
                                        int level,
                                        const std::filesystem::path& outputDir,
                                        const OrientedBBox& bbox,
                                        QString& errorMessage)
{
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        errorMessage = QObject::tr("No volume package loaded");
        return false;
    }
    if (!_cache) {
        errorMessage = QObject::tr("Chunk cache unavailable");
        return false;
    }

    auto volumePtr = volumePkg->volume(volumeId);
    if (!volumePtr) {
        errorMessage = QObject::tr("Volume %1 is not available").arg(QString::fromStdString(volumeId));
        return false;
    }

    z5::Dataset* dataset = volumePtr->zarrDataset(level);
    QString samplingError;
    const auto scaleFactors = volumePtr->levelScaleFactors(level);
    const auto translations = volumePtr->levelTranslations(level);

    if (dataset) {
        std::cout << "[Cutout] exportToTiff volume=" << volumeId
                  << " level=" << level
                  << " datasetShape=" << shapeToString(dataset->shape())
                  << " bboxCenter=" << vecToString(bbox.center)
                  << " bboxHalfExtents=" << vecToString(bbox.halfExtents)
                  << " axisU=" << vecToString(bbox.axisU)
                  << " axisV=" << vecToString(bbox.axisV)
                  << " axisN=" << vecToString(bbox.axisN)
                  << " scaleFactors=" << '(' << scaleFactors[0] << ',' << scaleFactors[1] << ',' << scaleFactors[2] << ')'
                  << " translations=" << '(' << translations[0] << ',' << translations[1] << ',' << translations[2] << ')'
                  << std::endl;
    } else {
        std::cout << "[Cutout] exportToTiff volume=" << volumeId
                  << " level=" << level
                  << " dataset=null" << std::endl;
    }

    auto blockOpt = sampleIntensityBlock(dataset, _cache, bbox, scaleFactors, translations, samplingError);
    if (!blockOpt) {
        errorMessage = samplingError;
        return false;
    }
    const VolumeBlock& block = *blockOpt;

    auto minmax = std::minmax_element(block.data.begin(), block.data.end());
    const auto nonZero = static_cast<size_t>(std::count_if(block.data.begin(), block.data.end(), [](uint8_t v) { return v != 0; }));
    std::cout << "[Cutout] sampled block size=" << vecToString(block.size)
              << " totalVoxels=" << block.data.size()
              << " min=" << static_cast<int>(*minmax.first)
              << " max=" << static_cast<int>(*minmax.second)
              << " nonZero=" << nonZero
              << std::endl;

    std::string fileStem = coordinateFileStem(dataset, bbox, scaleFactors, translations, level, "");
    if (fileStem.empty()) {
        fileStem = "bbox";
        if (level != 0) {
            fileStem += "_L" + std::to_string(level);
        }
    }
    std::filesystem::path tiffPath = outputDir / (fileStem + ".tif");

    if (std::filesystem::exists(tiffPath)) {
        errorMessage = QObject::tr("Output path %1 already exists")
            .arg(QString::fromStdString(tiffPath.string()));
        return false;
    }

    std::vector<cv::Mat> slices;
    slices.reserve(static_cast<size_t>(block.size[0]));
    size_t idx = 0;
    for (int z = 0; z < block.size[0]; ++z) {
        cv::Mat slice(block.size[1], block.size[2], CV_8UC1);
        for (int y = 0; y < block.size[1]; ++y) {
            uint8_t* dst = slice.ptr<uint8_t>(y);
            std::copy(block.data.begin() + idx, block.data.begin() + idx + block.size[2], dst);
            idx += block.size[2];
        }
        slices.emplace_back(std::move(slice));
    }

    if (!cv::imwritemulti(tiffPath.string(), slices)) {
        errorMessage = QObject::tr("Failed to write %1").arg(QString::fromStdString(tiffPath.string()));
        return false;
    }

    return true;
}


bool OverlayBoundingBox3d::exportSegmentationsToZarr(const std::vector<std::string>& segmentationIds,
                                                     const std::string& volumeId,
                                                     int level,
                                                     const std::filesystem::path& outputDir,
                                                     const OrientedBBox& bbox,
                                                     QString& errorMessage) const
{
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        errorMessage = QObject::tr("No volume package loaded");
        return false;
    }

    auto volumePtr = volumePkg->volume(volumeId);
    if (!volumePtr) {
        errorMessage = QObject::tr("Volume %1 is not available").arg(QString::fromStdString(volumeId));
        return false;
    }

    const double levelScale = std::pow(2.0, level);
    const auto scaleFactors = volumePtr->levelScaleFactors(level);
    const auto translations = volumePtr->levelTranslations(level);
    z5::Dataset* dataset = volumePtr->zarrDataset(level);

    std::map<std::string, QuadSurface*> surfaceMap;
    std::vector<std::unique_ptr<QuadSurface>> storage;
    if (!buildShiftedSurfaces(segmentationIds, bbox, static_cast<float>(levelScale), surfaceMap, storage, errorMessage)) {
        return false;
    }

    SamplingGrid grid = makeSamplingGrid(bbox);

    std::vector<uint8_t> volumeData;
    if (!voxelizeSegmentationsViewerStyle(surfaceMap, grid, volumeData)) {
        errorMessage = QObject::tr("Failed to voxelize segmentations for the selected bounding box");
        return false;
    }

    std::string fileStem = coordinateFileStem(dataset, bbox, scaleFactors, translations, level, "_seg");
    if (fileStem.empty()) {
        fileStem = "bbox_seg";
        if (level != 0) {
            fileStem += "_L" + std::to_string(level);
        }
    }
    std::filesystem::path outPath = outputDir / (fileStem + ".zarr");

    if (std::filesystem::exists(outPath)) {
        errorMessage = QObject::tr("Output path %1 already exists")
            .arg(QString::fromStdString(outPath.string()));
        return false;
    }

    if (!writeViewerStyleVolumeToZarr(grid, volumeData, outPath, errorMessage)) {
        return false;
    }

    const double voxSize = volumePtr->voxelSize();
    const float voxelSizeLevel = voxSize > 0.0 ? static_cast<float>(voxSize * levelScale) : 1.0f;

    try {
        z5::filesystem::handle::File outFile(outPath);

        nlohmann::json attrs;
        nlohmann::json surfacesJson = nlohmann::json::array();
        for (const auto& [segId, _] : surfaceMap) {
            surfacesJson.push_back(segId);
        }
        attrs["surfaces"] = surfacesJson;
        attrs["source_level"] = level;
        attrs["voxel_size_level"] = voxelSizeLevel;
        attrs["voxel_size"] = voxelSizeLevel;
        attrs["volume_dimensions"] = {grid.width, grid.height, grid.depth};

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        attrs["created"] = ss.str();

        nlohmann::json orientation;
        orientation["center"] = {bbox.center[0], bbox.center[1], bbox.center[2]};
        orientation["axis_u"] = {bbox.axisU[0], bbox.axisU[1], bbox.axisU[2]};
        orientation["axis_v"] = {bbox.axisV[0], bbox.axisV[1], bbox.axisV[2]};
        orientation["axis_n"] = {bbox.axisN[0], bbox.axisN[1], bbox.axisN[2]};
        orientation["half_extents"] = {bbox.halfExtents[0], bbox.halfExtents[1], bbox.halfExtents[2]};
        orientation["grid_shape"] = {grid.width, grid.height, grid.depth};
        orientation["grid_step"] = {grid.stepU, grid.stepV, grid.stepN};
        orientation["level"] = level;
        attrs["orientation"] = orientation;

        nlohmann::json multiscale;
        multiscale["version"] = "0.4";
        multiscale["name"] = "bbox_seg";
        multiscale["axes"] = nlohmann::json::array({
            nlohmann::json{{"name","z"},{"type","space"},{"unit","pixel"}},
            nlohmann::json{{"name","y"},{"type","space"},{"unit","pixel"}},
            nlohmann::json{{"name","x"},{"type","space"},{"unit","pixel"}}
        });

        nlohmann::json datasets = nlohmann::json::array();
        constexpr int kNumPyramidLevels = 5;
        for (int lvl = 0; lvl < kNumPyramidLevels; ++lvl) {
            double scale = std::pow(2.0, static_cast<double>(lvl));
            nlohmann::json entry;
            entry["path"] = std::to_string(lvl);
            entry["coordinateTransformations"] = nlohmann::json::array({
                nlohmann::json{{"type","scale"},{"scale", nlohmann::json::array({scale, scale, scale})}},
                nlohmann::json{{"type","translation"},{"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}
            });
            datasets.push_back(entry);
        }
        multiscale["datasets"] = datasets;
        multiscale["metadata"] = nlohmann::json{{"downsampling_method", "max"}};

        attrs["multiscales"] = nlohmann::json::array({multiscale});

        z5::filesystem::writeAttributes(outFile, attrs);
    } catch (const std::exception& e) {
        errorMessage = QObject::tr("Failed to annotate segmentation zarr: %1").arg(e.what());
        return false;
    }

    return true;
}


bool OverlayBoundingBox3d::exportSegmentationsToTiff(const std::vector<std::string>& segmentationIds,
                                                     const std::string& volumeId,
                                                     int level,
                                                     const std::filesystem::path& outputDir,
                                                     const OrientedBBox& bbox,
                                                     QString& errorMessage) const
{
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        errorMessage = QObject::tr("No volume package loaded");
        return false;
    }

    auto volumePtr = volumePkg->volume(volumeId);
    if (!volumePtr) {
        errorMessage = QObject::tr("Volume %1 is not available").arg(QString::fromStdString(volumeId));
        return false;
    }

    const double levelScale = std::pow(2.0, level);
    const auto scaleFactors = volumePtr->levelScaleFactors(level);
    const auto translations = volumePtr->levelTranslations(level);
    z5::Dataset* dataset = volumePtr->zarrDataset(level);

    std::map<std::string, QuadSurface*> surfaceMap;
    std::vector<std::unique_ptr<QuadSurface>> storage;
    if (!buildShiftedSurfaces(segmentationIds, bbox, static_cast<float>(levelScale), surfaceMap, storage, errorMessage)) {
        return false;
    }

    SamplingGrid grid = makeSamplingGrid(bbox);
    std::vector<uint8_t> volumeData;
    if (!voxelizeSegmentationsViewerStyle(surfaceMap, grid, volumeData)) {
        errorMessage = QObject::tr("Failed to voxelize segmentations for the selected bounding box");
        return false;
    }

    std::string fileStem = coordinateFileStem(dataset, bbox, scaleFactors, translations, level, "_seg");
    if (fileStem.empty()) {
        fileStem = "bbox_seg";
        if (level != 0) {
            fileStem += "_L" + std::to_string(level);
        }
    }
    std::filesystem::path tiffPath = outputDir / (fileStem + ".tif");

    if (std::filesystem::exists(tiffPath)) {
        errorMessage = QObject::tr("Output path %1 already exists")
            .arg(QString::fromStdString(tiffPath.string()));
        return false;
    }

    std::vector<cv::Mat> slices;
    const int depth = std::max(1, grid.depth);
    const int height = std::max(1, grid.height);
    const int width = std::max(1, grid.width);
    slices.reserve(static_cast<size_t>(depth));
    for (int z = 0; z < depth; ++z) {
        cv::Mat slice(height, width, CV_8UC1);
        const std::size_t base = static_cast<std::size_t>(z) * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        for (int y = 0; y < height; ++y) {
            const uint8_t* srcRow = volumeData.data() + base + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            uint8_t* dstRow = slice.ptr<uint8_t>(y);
            std::copy_n(srcRow, static_cast<std::size_t>(width), dstRow);
        }
        slices.emplace_back(std::move(slice));
    }

    if (!cv::imwritemulti(tiffPath.string(), slices)) {
        errorMessage = QObject::tr("Failed to write %1").arg(QString::fromStdString(tiffPath.string()));
        return false;
    }

    return true;
}


void OverlayBoundingBox3d::showCutOutDialog(QWidget* parent)
{
    if (!_sharedBox) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Draw a bounding box before exporting."));
        return;
    }

    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("No volume package loaded."));
        return;
    }

    const auto volumeIds = volumePkg->volumeIDs();
    if (volumeIds.empty()) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("This volume package has no volumes."));
        return;
    }

    int minLevels = std::numeric_limits<int>::max();
    bool haveLevelInfo = false;
    for (const auto& id : volumeIds) {
        if (auto volume = volumePkg->volume(id)) {
            size_t scales = volume->numScales();
            if (scales == 0) {
                scales = 1;
            }
            minLevels = std::min(minLevels, static_cast<int>(scales));
            haveLevelInfo = true;
        }
    }
    if (!haveLevelInfo) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Unable to determine volume resolutions."));
        return;
    }
    if (minLevels <= 0 || minLevels == std::numeric_limits<int>::max()) {
        minLevels = 1;
    }

    QDialog dialog(parent ? parent : &_window);
    dialog.setWindowTitle(QObject::tr("Cut Out BBox"));
    dialog.setModal(true);

    auto* layout = new QVBoxLayout(&dialog);

    const OrientedBBox& oriented = *_sharedBox;
    const Rect3D bounds = orientedBBoxToRect(oriented);
    const QString bboxText = QObject::tr("Bounds (xyz): [%1, %2, %3] – [%4, %5, %6]")
        .arg(QString::number(bounds.low[0], 'f', 2))
        .arg(QString::number(bounds.low[1], 'f', 2))
        .arg(QString::number(bounds.low[2], 'f', 2))
        .arg(QString::number(bounds.high[0], 'f', 2))
        .arg(QString::number(bounds.high[1], 'f', 2))
        .arg(QString::number(bounds.high[2], 'f', 2));
    layout->addWidget(new QLabel(bboxText, &dialog));

    const QString orientationText = QObject::tr("Center: [%1, %2, %3]  |  Half extents: [%4, %5, %6]")
        .arg(QString::number(oriented.center[0], 'f', 2))
        .arg(QString::number(oriented.center[1], 'f', 2))
        .arg(QString::number(oriented.center[2], 'f', 2))
        .arg(QString::number(oriented.halfExtents[0], 'f', 2))
        .arg(QString::number(oriented.halfExtents[1], 'f', 2))
        .arg(QString::number(oriented.halfExtents[2], 'f', 2));
    layout->addWidget(new QLabel(orientationText, &dialog));

    const QString axesText = QObject::tr("Axes U:%1  V:%2  N:%3")
        .arg(QString(" [%1, %2, %3]")
                 .arg(QString::number(oriented.axisU[0], 'f', 2))
                 .arg(QString::number(oriented.axisU[1], 'f', 2))
                 .arg(QString::number(oriented.axisU[2], 'f', 2)))
        .arg(QString(" [%1, %2, %3]")
                 .arg(QString::number(oriented.axisV[0], 'f', 2))
                 .arg(QString::number(oriented.axisV[1], 'f', 2))
                 .arg(QString::number(oriented.axisV[2], 'f', 2)))
        .arg(QString(" [%1, %2, %3]")
                 .arg(QString::number(oriented.axisN[0], 'f', 2))
                 .arg(QString::number(oriented.axisN[1], 'f', 2))
                 .arg(QString::number(oriented.axisN[2], 'f', 2)));
    layout->addWidget(new QLabel(axesText, &dialog));

    layout->addWidget(new QLabel(QObject::tr("Output format:"), &dialog));
    auto* formatGroup = new QButtonGroup(&dialog);
    auto* radioZarr = new QRadioButton(QObject::tr("OME-Zarr (6 levels, 2× downsampled)"), &dialog);
    auto* radioTiff = new QRadioButton(QObject::tr("3D TIFF (uint8)"), &dialog);
    radioTiff->setChecked(true);
    formatGroup->addButton(radioZarr, 0);
    formatGroup->addButton(radioTiff, 1);
    layout->addWidget(radioZarr);
    layout->addWidget(radioTiff);

    auto* levelLayout = new QHBoxLayout();
    levelLayout->addWidget(new QLabel(QObject::tr("Source resolution:"), &dialog));
    auto* levelCombo = new QComboBox(&dialog);
    for (int level = 0; level < minLevels; ++level) {
        unsigned long long factor = 1ULL << level;
        QString detail = level == 0 ? QObject::tr("native")
                                    : QObject::tr("%1× downsample").arg(static_cast<qulonglong>(factor));
        levelCombo->addItem(QObject::tr("Level %1 (%2)").arg(level).arg(detail), level);
    }
    levelLayout->addWidget(levelCombo);
    layout->addLayout(levelLayout);

    auto* intensityCheck = new QCheckBox(QObject::tr("Export intensity volume"), &dialog);
    intensityCheck->setChecked(true);
    layout->addWidget(intensityCheck);

    auto* segmentationCheck = new QCheckBox(QObject::tr("Export segmentation volume"), &dialog);
    layout->addWidget(segmentationCheck);

    auto* volumesGroup = new QGroupBox(QObject::tr("Volumes"), &dialog);
    auto* volumesLayout = new QVBoxLayout(volumesGroup);
    struct VolumeOption { std::string id; QCheckBox* checkbox; };
    std::vector<VolumeOption> volumeOptions;
    volumeOptions.reserve(volumeIds.size());
    for (const auto& id : volumeIds) {
        auto volumePtr = volumePkg->volume(id);
        if (!volumePtr) {
            continue;
        }
        QString labelText = QString::fromStdString(id);
        const std::string& volName = volumePtr->name();
        if (!volName.empty()) {
            labelText = QStringLiteral("%1 (%2)").arg(QString::fromStdString(id), QString::fromStdString(volName));
        }
        auto* check = new QCheckBox(labelText, volumesGroup);
        check->setChecked(true);
        volumesLayout->addWidget(check);
        volumeOptions.push_back({id, check});
    }

    if (volumeOptions.empty()) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("No loadable volumes found."));
        return;
    }

    layout->addWidget(volumesGroup);

    layout->addWidget(new QLabel(QObject::tr("Output folder:"), &dialog));
    auto* dirLayout = new QHBoxLayout();
    auto* dirEdit = new QLineEdit(&dialog);
    QString defaultDir = QString::fromStdString(volumePkg->getVolpkgDirectory());
    if (defaultDir.isEmpty()) {
        defaultDir = QDir::homePath();
    }
    dirEdit->setText(defaultDir);
    auto* browseButton = new QPushButton(QObject::tr("Browse..."), &dialog);
    dirLayout->addWidget(dirEdit);
    dirLayout->addWidget(browseButton);
    layout->addLayout(dirLayout);

    QObject::connect(browseButton, &QPushButton::clicked, &dialog, [dirEdit, &dialog]() {
        const QString selected = QFileDialog::getExistingDirectory(&dialog,
            QObject::tr("Select Output Folder"),
            dirEdit->text());
        if (!selected.isEmpty()) {
            dirEdit->setText(selected);
        }
    });

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    QVariant levelData = levelCombo->currentData();
    const int selectedLevel = levelData.isValid() ? levelData.toInt() : 0;

    const bool exportIntensity = intensityCheck->isChecked();
    const bool exportSegmentation = segmentationCheck->isChecked();

    if (!exportIntensity && !exportSegmentation) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Select at least one export option."));
        return;
    }

    std::vector<std::string> selectedVolumes;
    for (const auto& opt : volumeOptions) {
        if (opt.checkbox && opt.checkbox->isChecked()) {
            selectedVolumes.push_back(opt.id);
        }
    }
    if (selectedVolumes.empty()) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Select at least one volume."));
        return;
    }

    const QString outputDirStr = dirEdit->text().trimmed();
    if (outputDirStr.isEmpty()) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Choose an output folder."));
        return;
    }

    const std::filesystem::path outputDir = outputDirStr.toStdString();
    try {
        if (!std::filesystem::exists(outputDir)) {
            std::filesystem::create_directories(outputDir);
        } else if (!std::filesystem::is_directory(outputDir)) {
            QMessageBox::warning(parent ? parent : &_window,
                                 QObject::tr("Cut Out BBox"),
                                 QObject::tr("%1 is not a directory.").arg(outputDirStr));
            return;
        }
    } catch (const std::exception& e) {
        QMessageBox::warning(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             QObject::tr("Failed to prepare output folder: %1").arg(e.what()));
        return;
    }

    const ExportFormat format = formatGroup->checkedId() == 1 ? ExportFormat::Tiff : ExportFormat::Zarr;

    QString errorMessage;
    int intensityCount = 0;
    int segmentationCount = 0;

    std::vector<std::string> segmentationIds;
    if (exportSegmentation) {
        segmentationIds = segmentationsWithin(*_sharedBox);
        if (segmentationIds.empty()) {
            QMessageBox::warning(parent ? parent : &_window,
                                 QObject::tr("Cut Out BBox"),
                                 QObject::tr("No segmentations intersect the current bounding box."));
            return;
        }
    }

    for (const auto& volumeId : selectedVolumes) {
        if (exportIntensity) {
            const bool ok = (format == ExportFormat::Zarr)
                ? exportToZarr(volumeId, selectedLevel, outputDir, *_sharedBox, errorMessage)
                : exportToTiff(volumeId, selectedLevel, outputDir, *_sharedBox, errorMessage);
            if (!ok) {
                QMessageBox::warning(parent ? parent : &_window,
                                     QObject::tr("Cut Out BBox"),
                                     errorMessage);
                return;
            }
            ++intensityCount;
        }

        if (exportSegmentation) {
            const bool okSeg = (format == ExportFormat::Zarr)
                ? exportSegmentationsToZarr(segmentationIds, volumeId, selectedLevel, outputDir, *_sharedBox, errorMessage)
                : exportSegmentationsToTiff(segmentationIds, volumeId, selectedLevel, outputDir, *_sharedBox, errorMessage);
            if (!okSeg) {
                QMessageBox::warning(parent ? parent : &_window,
                                     QObject::tr("Cut Out BBox"),
                                     errorMessage);
                return;
            }
            ++segmentationCount;
        }
    }

    QStringList summaryParts;
    const QString formatLabel = format == ExportFormat::Zarr ? QObject::tr("OME-Zarr") : QObject::tr("TIFF");
    if (exportIntensity) {
        summaryParts << QObject::tr("%1 intensity %2 (level %3)")
                            .arg(intensityCount)
                            .arg(formatLabel)
                            .arg(selectedLevel);
    }
    if (exportSegmentation) {
        summaryParts << QObject::tr("%1 segmentation %2 (level %3)")
                            .arg(segmentationCount)
                            .arg(formatLabel)
                            .arg(selectedLevel);
    }

    if (auto* bar = _window.statusBar()) {
        const QString summaryText = summaryParts.isEmpty() ? QObject::tr("no files") : summaryParts.join(", ");
        bar->showMessage(
            QObject::tr("Saved %1 to %2")
                .arg(summaryText)
                .arg(outputDirStr),
            STATUS_TIMEOUT_MS);
    }

    QString infoText = QObject::tr("Export complete (level %1).").arg(selectedLevel);
    if (!summaryParts.isEmpty()) {
        infoText += QStringLiteral(" \u2014 ") + summaryParts.join(", ");
    }
    QMessageBox::information(parent ? parent : &_window,
                             QObject::tr("Cut Out BBox"),
                             infoText);
}

std::vector<std::string> OverlayBoundingBox3d::segmentationsWithin(const OrientedBBox& bbox) const
{
    std::vector<std::string> matches;
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        return matches;
    }

    const Rect3D query = normalizeRect(orientedBBoxToRect(bbox));

    for (const auto& segId : volumePkg->segmentationIDs()) {
        try {
            auto surfMeta = volumePkg->getSurface(segId);
            if (!surfMeta) {
                continue;
            }

            Rect3D segBox = normalizeRect(surfMeta->bbox);
            bool degenerate = false;
            for (int axis = 0; axis < 3; ++axis) {
                if (segBox.high[axis] < segBox.low[axis]) {
                    degenerate = true;
                    break;
                }
            }

            if (degenerate) {
                if (auto* quad = dynamic_cast<QuadSurface*>(surfMeta->surface())) {
                    segBox = normalizeRect(quad->bbox());
                    degenerate = false;
                    for (int axis = 0; axis < 3; ++axis) {
                        if (segBox.high[axis] < segBox.low[axis]) {
                            degenerate = true;
                            break;
                        }
                    }
                }
            }

            if (!degenerate && intersect(segBox, query)) {
                matches.push_back(segId);
            }
        } catch (const std::exception&) {
            // Skip segmentations that fail to load
        }
    }

    return matches;
}

std::vector<std::string> OverlayBoundingBox3d::segmentationsInCurrentBBox() const
{
    if (!_sharedBox) {
        return {};
    }
    return segmentationsWithin(*_sharedBox);
}

bool OverlayBoundingBox3d::buildShiftedSurfaces(const std::vector<std::string>& segmentationIds,
                                                const OrientedBBox& bbox,
                                                float levelScale,
                                                std::map<std::string, QuadSurface*>& outSurfaces,
                                                std::vector<std::unique_ptr<QuadSurface>>& storage,
                                                QString& errorMessage) const
{
    auto volumePkg = _volumePkg.lock();
    if (!volumePkg) {
        errorMessage = QObject::tr("No volume package loaded");
        return false;
    }

    storage.clear();
    outSurfaces.clear();
    storage.reserve(segmentationIds.size());

    if (levelScale <= 0.f) {
        levelScale = 1.f;
    }

    const float invScale = 1.f / levelScale;
    SamplingGrid grid = makeSamplingGrid(bbox);

    const cv::Vec3f axisU = normalizeVec(bbox.axisU);
    const cv::Vec3f axisV = normalizeVec(bbox.axisV);
    const cv::Vec3f axisN = normalizeVec(bbox.axisN);
    const cv::Vec3f centerScaled = bbox.center * invScale;

    const float halfUScaled = bbox.halfExtents[0] * invScale;
    const float halfVScaled = bbox.halfExtents[1] * invScale;
    const float halfNScaled = bbox.halfExtents[2] * invScale;

    const float stepUScaled = (grid.width > 1 && halfUScaled > 1e-4f)
        ? (2.f * halfUScaled) / static_cast<float>(grid.width - 1)
        : 0.f;
    const float stepVScaled = (grid.height > 1 && halfVScaled > 1e-4f)
        ? (2.f * halfVScaled) / static_cast<float>(grid.height - 1)
        : 0.f;
    const float stepNScaled = (grid.depth > 1 && halfNScaled > 1e-4f)
        ? (2.f * halfNScaled) / static_cast<float>(grid.depth - 1)
        : 0.f;

    for (const auto& segId : segmentationIds) {
        try {
            auto surfMeta = volumePkg->getSurface(segId);
            if (!surfMeta) {
                continue;
            }

            QuadSurface* source = surfMeta->surface();
            if (!source) {
                continue;
            }

            cv::Mat_<cv::Vec3f> original = source->rawPoints();
            cv::Mat_<cv::Vec3f> shifted(original.rows, original.cols, cv::Vec3f(-1.f, -1.f, -1.f));
            bool hasAny = false;

            for (int r = 0; r < original.rows; ++r) {
                for (int c = 0; c < original.cols; ++c) {
                    const cv::Vec3f& p = original(r, c);
                    if (p[0] < 0.f) {
                        continue;
                    }

                    cv::Vec3f pScaled = p * invScale;
                    cv::Vec3f rel = pScaled - centerScaled;
                    float u = rel.dot(axisU);
                    float v = rel.dot(axisV);
                    float n = rel.dot(axisN);

                    if (halfUScaled > 1e-4f && (u < -halfUScaled || u > halfUScaled)) {
                        continue;
                    }
                    if (halfVScaled > 1e-4f && (v < -halfVScaled || v > halfVScaled)) {
                        continue;
                    }
                    if (halfNScaled > 1e-4f && (n < -halfNScaled || n > halfNScaled)) {
                        continue;
                    }

                    float x = (grid.width <= 1 || stepUScaled < 1e-6f)
                        ? static_cast<float>(grid.width - 1) * 0.5f
                        : gridIndexForValue(u, grid.width, halfUScaled, stepUScaled);
                    float y = (grid.height <= 1 || stepVScaled < 1e-6f)
                        ? static_cast<float>(grid.height - 1) * 0.5f
                        : gridIndexForValue(v, grid.height, halfVScaled, stepVScaled);
                    float z = (grid.depth <= 1 || stepNScaled < 1e-6f)
                        ? static_cast<float>(grid.depth - 1) * 0.5f
                        : gridIndexForValue(n, grid.depth, halfNScaled, stepNScaled);

                    if (x < -0.5f || y < -0.5f || z < -0.5f ||
                        x > static_cast<float>(grid.width) - 0.5f ||
                        y > static_cast<float>(grid.height) - 0.5f ||
                        z > static_cast<float>(grid.depth) - 0.5f) {
                        continue;
                    }

                    shifted(r, c) = cv::Vec3f(x, y, z);
                    hasAny = true;
                }
            }

            if (!hasAny) {
                continue;
            }

            auto clone = std::make_unique<QuadSurface>(shifted, source->scale());
            outSurfaces.emplace(segId, clone.get());
            storage.push_back(std::move(clone));
        } catch (const std::exception& e) {
            errorMessage = QObject::tr("Failed to prepare segmentation %1: %2")
                .arg(QString::fromStdString(segId))
                .arg(e.what());
            return false;
        }
    }

    if (outSurfaces.empty()) {
        errorMessage = QObject::tr("No segmentations could be prepared for voxelization");
        return false;
    }

    return true;
}
