#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QCursor>
#include <QLineF>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"
#include "COutlinedTextItem.hpp"
#include "OverlaySegmentationIntersections.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>

#include "OpChain.hpp"
#include "vc/core/util/Render.hpp"

using qga = QGuiApplication;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
// More gentle zoom factor for smoother experience
#define ZOOM_FACTOR 1.05 // Reduced from 1.15 for even smoother zooming

#define COLOR_CURSOR Qt::cyan
#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

constexpr float MIN_ZOOM = 0.03125f;
constexpr float MAX_ZOOM = 4.0f;

#include <limits>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cmath>

namespace {

std::string vecToString(const cv::Vec3f& v)
{
    std::ostringstream oss;
    oss << '(' << v[0] << ',' << v[1] << ',' << v[2] << ')';
    return oss.str();
}

constexpr float kPi = 3.14159265358979323846f;

float normalizeAngle(float angle)
{
    const float twoPi = 2.f * kPi;
    while (angle > kPi) {
        angle -= twoPi;
    }
    while (angle < -kPi) {
        angle += twoPi;
    }
    return angle;
}

std::optional<float> rotationAngleForPoint(const cv::Vec3f& center,
                                           const cv::Vec3f& point,
                                           const cv::Vec3f& axisPrimary,
                                           const cv::Vec3f& axisSecondary,
                                           const cv::Vec3f& axisNormal)
{
    cv::Vec3f rel = point - center;
    float projN = rel.dot(axisNormal);
    cv::Vec3f relPlane = rel - axisNormal * projN;
    float len = cv::norm(relPlane);
    if (len <= 1e-4f) {
        return std::nullopt;
    }
    cv::Vec3f dir = relPlane / len;
    float cosTheta = dir.dot(axisPrimary);
    float sinTheta = dir.dot(axisSecondary);
    if (!std::isfinite(cosTheta) || !std::isfinite(sinTheta)) {
        return std::nullopt;
    }
    return std::atan2(sinTheta, cosTheta);
}

} // namespace
#include <cmath>

namespace {
constexpr float BBOX_HANDLE_RADIUS = 8.0f;
constexpr float BBOX_HANDLE_PICK_RADIUS = 12.0f;
constexpr float BBOX_HANDLE_DIAMETER = BBOX_HANDLE_RADIUS * 2.0f;
constexpr float BBOX_ROTATION_HANDLE_RADIUS = 7.0f;
constexpr float BBOX_ROTATION_HANDLE_DIAMETER = BBOX_ROTATION_HANDLE_RADIUS * 2.0f;
constexpr float BBOX_ROTATION_HANDLE_OFFSET = 30.0f;

inline cv::Vec3f bboxAxisByIndex(const OrientedBBox& box, int index)
{
    switch (index) {
    case 0: return box.axisU;
    case 1: return box.axisV;
    default: return box.axisN;
    }
}

inline float bboxHalfExtentByIndex(const OrientedBBox& box, int index)
{
    switch (index) {
    case 0: return box.halfExtents[0];
    case 1: return box.halfExtents[1];
    default: return box.halfExtents[2];
    }
}

inline float& bboxHalfExtentRefByIndex(OrientedBBox& box, int index)
{
    switch (index) {
    case 0: return box.halfExtents[0];
    case 1: return box.halfExtents[1];
    default: return box.halfExtents[2];
    }
}

inline void setBBoxAxisByIndex(OrientedBBox& box, int index, const cv::Vec3f& axis)
{
    switch (index) {
    case 0: box.axisU = axis; break;
    case 1: box.axisV = axis; break;
    default: box.axisN = axis; break;
    }
}

inline cv::Vec3f normalizedAxis(const cv::Vec3f& axis)
{
    float norm = cv::norm(axis);
    if (norm <= 1e-6f) {
        return axis;
    }
    return axis / norm;
}

inline std::optional<std::array<int,3>> mapDatasetAxesToBoxAxes(const OrientedBBox& box)
{
    std::array<int,3> mapping{ -1, -1, -1 };
    std::array<bool,3> axisUsed{ false, false, false };
    constexpr float kTolerance = 0.5f;

    for (int datasetAxis = 0; datasetAxis < 3; ++datasetAxis) {
        const cv::Vec3f target = axisIndexToUnit(datasetAxis);
        int bestIndex = -1;
        float bestDot = -1.0f;
        for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
            if (axisUsed[axisIdx]) {
                continue;
            }
            const cv::Vec3f axisVec = bboxAxisByIndex(box, axisIdx);
            const float dot = std::abs(axisVec.dot(target));
            if (dot > bestDot) {
                bestDot = dot;
                bestIndex = axisIdx;
            }
        }
        if (bestIndex < 0 || bestDot < kTolerance) {
            return std::nullopt;
        }
        mapping[datasetAxis] = bestIndex;
        axisUsed[bestIndex] = true;
    }

    return mapping;
}

inline float bboxSupportAlongNormal(const OrientedBBox& box, const cv::Vec3f& normal)
{
    const cv::Vec3f unitNormal = normalizedAxis(normal);
    if (!std::isfinite(unitNormal[0]) || !std::isfinite(unitNormal[1]) || !std::isfinite(unitNormal[2])) {
        return 0.0f;
    }

    const cv::Vec3f axisU = normalizedAxis(bboxAxisByIndex(box, 0));
    const cv::Vec3f axisV = normalizedAxis(bboxAxisByIndex(box, 1));
    const cv::Vec3f axisN = normalizedAxis(bboxAxisByIndex(box, 2));

    const float projU = std::abs(axisU.dot(unitNormal)) * box.halfExtents[0];
    const float projV = std::abs(axisV.dot(unitNormal)) * box.halfExtents[1];
    const float projN = std::abs(axisN.dot(unitNormal)) * box.halfExtents[2];
    return projU + projV + projN;
}
}

// Helper: remove spatial outliers based on robust neighbor-distance stats
static cv::Mat_<cv::Vec3f> clean_surface_outliers(const cv::Mat_<cv::Vec3f>& points, float distance_threshold = 5.0f)
{
    cv::Mat_<cv::Vec3f> cleaned = points.clone();

    std::vector<float> all_neighbor_dists;
    all_neighbor_dists.reserve(points.rows * points.cols);

    // First pass: gather neighbor distances
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            const cv::Vec3f center = points(j, i);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int ny = j + dy;
                    const int nx = i + dx;
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            const cv::Vec3f neighbor = points(ny, nx);
                            float dist = cv::norm(center - neighbor);
                            if (std::isfinite(dist) && dist > 0) {
                                all_neighbor_dists.push_back(dist);
                            }
                        }
                    }
                }
            }
        }
    }

    float median_dist = 0.0f;
    float mad = 0.0f;
    if (!all_neighbor_dists.empty()) {
        std::sort(all_neighbor_dists.begin(), all_neighbor_dists.end());
        median_dist = all_neighbor_dists[all_neighbor_dists.size() / 2];
        std::vector<float> abs_devs;
        abs_devs.reserve(all_neighbor_dists.size());
        for (float d : all_neighbor_dists) abs_devs.push_back(std::abs(d - median_dist));
        std::sort(abs_devs.begin(), abs_devs.end());
        mad = abs_devs[abs_devs.size() / 2];
    }
    const float threshold = median_dist + distance_threshold * (mad / 0.6745f);

    // Second pass: invalidate isolated/far points
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            const cv::Vec3f center = points(j, i);
            float min_neighbor = std::numeric_limits<float>::infinity();
            int neighbor_count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int ny = j + dy;
                    const int nx = i + dx;
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            float dist = cv::norm(center - points(ny, nx));
                            if (std::isfinite(dist)) {
                                min_neighbor = std::min(min_neighbor, dist);
                                neighbor_count++;
                            }
                        }
                    }
                }
            }
            if (neighbor_count == 0 || (min_neighbor > threshold && threshold > 0)) {
                cleaned(j, i) = cv::Vec3f(-1.f, -1.f, -1.f);
            }
        }
    }
    return cleaned;
}


CVolumeViewer::CVolumeViewer(CSurfaceCollection *col, QWidget* parent)
    : QWidget(parent)
    , fGraphicsView(nullptr)
    , fBaseImageItem(nullptr)
    , _surf_col(col)
    , _highlighted_point_id(0)
    , _selected_point_id(0)
    , _dragged_point_id(0)
{
    // Create graphics view
    fGraphicsView = new CVolumeViewerView(this);
    
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    // setFocusProxy(fGraphicsView);
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CVolumeViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendResized, this, &CVolumeViewer::onResized);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CVolumeViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease, this, &CVolumeViewer::onPanRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendPanStart, this, &CVolumeViewer::onPanStart);
    connect(fGraphicsView, &CVolumeViewerView::sendMousePress, this, &CVolumeViewer::onMousePress);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseMove, this, &CVolumeViewer::onMouseMove);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseRelease, this, &CVolumeViewer::onMouseRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease, this, &CVolumeViewer::onKeyRelease);

    // Create graphics scene
    fScene = new QGraphicsScene({-2500,-2500,5000,5000}, this);

    // Set the scene
    fGraphicsView->setScene(fScene);

    QSettings settings("VC.ini", QSettings::IniFormat);
    // fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    // fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();
    _downscale_override = settings.value("perf/downscale_override", 0).toInt();
    _useFastInterpolation = settings.value("perf/fast_interpolation", false).toBool();
    if (_useFastInterpolation) {
        std::cout << "using nearest neighbor interpolation" << std::endl;
    }
    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);

    setLayout(aWidgetLayout);

    _overlayUpdateTimer = new QTimer(this);
    _overlayUpdateTimer->setSingleShot(true);
    _overlayUpdateTimer->setInterval(50);
    connect(_overlayUpdateTimer, &QTimer::timeout, this, &CVolumeViewer::updateAllOverlays);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : white; }");
    _lbl->move(10,5);
}

// Destructor
CVolumeViewer::~CVolumeViewer(void)
{
    delete fGraphicsView;
    delete fScene;
}

void round_scale(float &scale)
{
    if (abs(scale-round(log2(scale))) < 0.02f)
        scale = pow(2,round(log2(scale)));
    // the most reduced OME zarr projection is 32x so make the min zoom out 1/32 = 0.03125
    if (scale < MIN_ZOOM) scale = MIN_ZOOM;
    if (scale > MAX_ZOOM) scale = MAX_ZOOM;
}

//get center of current visible area in scene coordinates
QPointF visible_center(QGraphicsView *view)
{
    QRectF bbox = view->mapToScene(view->viewport()->geometry()).boundingRect();
    return bbox.topLeft() + QPointF(bbox.width(),bbox.height())*0.5;
}


QPointF CVolumeViewer::volumeToScene(const cv::Vec3f& vol_point)
{
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(_surf);
    QuadSurface* quad = dynamic_cast<QuadSurface*>(_surf);
    cv::Vec3f p;

    if (plane) {
        p = plane->project(vol_point, 1.0, _scale);
    } else if (quad) {
        auto ptr = quad->pointer();
        _surf->pointTo(ptr, vol_point, 4.0, 100);
        p = _surf->loc(ptr) * _scale;
    }

    return QPointF(p[0], p[1]);
}

bool scene2vol(cv::Vec3f &p, cv::Vec3f &n, Surface *_surf, const std::string &_surf_name, CSurfaceCollection *_surf_col, const QPointF &scene_loc, const cv::Vec2f &_vis_center, float _ds_scale)
{
    // Safety check for null surface
    if (!_surf) {
        p = cv::Vec3f(0, 0, 0);
        n = cv::Vec3f(0, 0, 1);
        return false;
    }
    
    try {
        cv::Vec3f surf_loc = {scene_loc.x()/_ds_scale, scene_loc.y()/_ds_scale,0};
        
        auto ptr = _surf->pointer();
        
        n = _surf->normal(ptr, surf_loc);
        p = _surf->coord(ptr, surf_loc);
    } catch (const cv::Exception& e) {
        return false;
    }
    return true;
}

void CVolumeViewer::onCursorMove(QPointF scene_loc)
{
    if (!_surf || !_surf_col)
        return;

    cv::Vec3f p, n;
    if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        if (_cursor) _cursor->hide();
    } else {
        if (_cursor) {
            _cursor->show();
            // Update cursor position visually without POI
            PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
            QuadSurface *quad = dynamic_cast<QuadSurface*>(_surf);
            cv::Vec3f sp;

            if (plane) {
                sp = plane->project(p, 1.0, _scale);
            } else if (quad) {
                auto ptr = quad->pointer();
                _surf->pointTo(ptr, p, 4.0, 100);
                sp = _surf->loc(ptr) * _scale;
            }
            _cursor->setPos(sp[0], sp[1]);
        }

        POI *cursor = _surf_col->poi("cursor");
        if (!cursor)
            cursor = new POI;
        cursor->p = p;
        _surf_col->setPOI("cursor", cursor);
    }

    if (_point_collection && _dragged_point_id == 0) {
        uint64_t old_highlighted_id = _highlighted_point_id;
        _highlighted_point_id = 0;

        const float highlight_dist_threshold = 10.0f;
        float min_dist_sq = highlight_dist_threshold * highlight_dist_threshold;

        for (const auto& item_pair : _points_items) {
            auto item = item_pair.second.circle;
            QPointF point_scene_pos = item->rect().center();
            QPointF diff = scene_loc - point_scene_pos;
            float dist_sq = QPointF::dotProduct(diff, diff);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                _highlighted_point_id = item_pair.first;
            }
        }

        if (old_highlighted_id != _highlighted_point_id) {
            if (auto old_point = _point_collection->getPoint(old_highlighted_id)) {
                renderOrUpdatePoint(*old_point);
            }
            if (auto new_point = _point_collection->getPoint(_highlighted_point_id)) {
                renderOrUpdatePoint(*new_point);
            }
        }
    }

    if (_bboxMode && isBBoxPlaneView()) {
        updateBBoxOverlay3D();
    }
}

void CVolumeViewer::recalcScales()
{
    float old_ds = _ds_scale;         // remember previous level
    // if (dynamic_cast<PlaneSurface*>(_surf))
    _min_scale = pow(2.0,1.-volume->numScales());
    // else
        // _min_scale = std::max(pow(2.0,1.-volume->numScales()), 0.5);
    
    /* -------- chooses _ds_scale/_ds_sd_idx -------- */
    if      (_scale >= _max_scale) { _ds_sd_idx = 0;                         }
    else if (_scale <  _min_scale) { _ds_sd_idx = volume->numScales()-1;     }
    else  { _ds_sd_idx = int(std::round(-std::log2(_scale))); }
    if (_downscale_override > 0) {
        _ds_sd_idx += _downscale_override;
        // Clamp to available scales
        _ds_sd_idx = std::min(_ds_sd_idx, (int)volume->numScales() - 1);
    }
    _ds_scale = std::pow(2.0f, -_ds_sd_idx);
    /* ---------------------------------------------------------------- */

    /* ---- refresh physical voxel size when pyramid level flips -- */
    if (volume && std::abs(_ds_scale - old_ds) > 1e-6f)
    {
        double vs = volume->voxelSize() / _ds_scale;   // µm per scene-unit
        fGraphicsView->setVoxelSize(vs, vs);           // keep scalebar honest
    }
}


void CVolumeViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    if (!_surf)
        return;

    if (_segmentationOverlay) {
        _segmentationOverlay->hideViewerItems(*this);
    }

    if (modifiers & Qt::ShiftModifier) {
        // Z slice navigation
        int adjustedSteps = steps;
        if (_surf_name == "segmentation") {
            adjustedSteps = (steps > 0) ? 1 : -1;
        }

        _z_off += adjustedSteps;

        // Clamp to valid range if we have volume data
        if (volume && dynamic_cast<PlaneSurface*>(_surf)) {
            PlaneSurface* plane = dynamic_cast<PlaneSurface*>(_surf);
            float effective_z = plane->origin()[2] + _z_off;
            effective_z = std::max(0.0f, std::min(effective_z, static_cast<float>(volume->numSlices() - 1)));
            _z_off = effective_z - plane->origin()[2];
        }

        renderVisible(true);
    }
    else {
        float zoom = pow(ZOOM_FACTOR, steps);
        _scale *= zoom;
        round_scale(_scale);
        //we should only zoom when we haven't hit the max / min, otherwise the zoom starts to pan center on the mouse
        if (_scale > MIN_ZOOM && _scale < MAX_ZOOM) {
            recalcScales();

            // The above scale is *not* part of Qt's scene-to-view transform, but part of the voxel-to-scene transform
            // implemented in PlaneSurface::project; it causes a zoom around the surface origin
            // Translations are represented in the Qt scene-to-view transform; these move the surface origin within the viewpoint
            // To zoom centered on the mouse, we adjust the scene-to-view translation appropriately
            // If the mouse were at the plane/surface origin, this adjustment should be zero
            // If the mouse were right of the plane origin, should translate to the left so that point ends up where it was
            fGraphicsView->translate(scene_loc.x() * (1 - zoom),
                                    scene_loc.y() * (1 - zoom));

            curr_img_area = {0,0,0,0};
            int max_size = 100000;
            fGraphicsView->setSceneRect(-max_size/2, -max_size/2, max_size, max_size);

        }
        renderVisible();
        updateSelectionGraphics();
    }

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();

    if (_bboxMode && isBBoxPlaneView()) {
        updateBBoxOverlay3D();
        updateBBoxCursor(scene_loc);
    }
}

void CVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> volume_)
{
    volume = volume_;
    
    // printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());

    int max_size = 100000 ;//std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*_ds_scale + 512;
    // printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
    
    if (volume->numScales() >= 2) {
        //FIXME currently hardcoded
        _max_scale = 0.5;
        _min_scale = pow(2.0,1.-volume->numScales());
    }
    else {
        //FIXME currently hardcoded
        _max_scale = 1.0;
        _min_scale = 1.0;
    }
    
    recalcScales();

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));

    renderVisible(true);

    // ——— Scalebar: physical size per scene-unit, compensating for down-sampling ———
    // volume->voxelSize() is µm per original voxel;
    // each scene-unit is still one original voxel, but we read data at (_ds_scale) resolution,
    // so we scale the voxelSize by 1/_ds_scale.
    double vs = volume->voxelSize() / _ds_scale;
    fGraphicsView->setVoxelSize(vs, vs);

    if (_bboxMode && isBBoxPlaneView()) {
        updateBBoxOverlay3D();
    }
}

void CVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (!_surf)
        return;

    // If a point was being dragged, don't do anything on release
    if (_dragged_point_id != 0) {
        return;
    }

    cv::Vec3f p, n;
    if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale))
        return;

    if (buttons == Qt::LeftButton) {
        bool isShift = modifiers.testFlag(Qt::ShiftModifier);

        if (isShift) {
            // If a collection is selected, add to it.
            if (_selected_collection_id != 0) {
                const auto& collections = _point_collection->getAllCollections();
                auto it = collections.find(_selected_collection_id);
                if (it != collections.end()) {
                    _point_collection->addPoint(it->second.name, p);
                }
            } else {
                // Otherwise, create a new collection.
                std::string new_name = _point_collection->generateNewCollectionName("col");
                auto new_point = _point_collection->addPoint(new_name, p);
                _selected_collection_id = new_point.collectionId;
                emit sendCollectionSelected(_selected_collection_id);
            }
        } else if (_highlighted_point_id != 0) {
            emit pointClicked(_highlighted_point_id);
        }
    }

    // Forward the click for focus
    if (dynamic_cast<PlaneSurface*>(_surf))
        sendVolumeClicked(p, n, _surf, buttons, modifiers);
    else if (_surf_name == "segmentation")
        sendVolumeClicked(p, n, _surf_col->surface("segmentation"), buttons, modifiers);
    else
        std::cout << "FIXME: onVolumeClicked()" << std::endl;
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::setPointCollection(VCCollection* point_collection)
{
    if (_point_collection) {
        disconnect(_point_collection, &VCCollection::collectionChanged, this, &CVolumeViewer::onCollectionChanged);
    }
    _point_collection = point_collection;
    if (_point_collection) {
        connect(_point_collection, &VCCollection::collectionChanged, this, &CVolumeViewer::onCollectionChanged);
    }
}

void CVolumeViewer::setSurface(const std::string &name)
{
    _surf_name = name;
    _surf = nullptr;
    onSurfaceChanged(name, _surf_col->surface(name));
}


void CVolumeViewer::invalidateVis()
{
    _slice_vis_valid = false;    
    for(auto &item : slice_vis_items) {
        fScene->removeItem(item);
        delete item;
    }
    slice_vis_items.resize(0);
}

void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    if (!_segmentationOverlay) {
        return;
    }

    if (name.empty() || name == _surf_name) {
        _segmentationOverlay->invalidateViewer(*this);
    } else {
        _segmentationOverlay->invalidateViewer(*this, name);
    }
}


void CVolumeViewer::onIntersectionChanged(std::string a, std::string b, Intersection *intersection)
{
    if (_segmentationOverlay) {
        _segmentationOverlay->handleIntersectionChanged(*this, a, b, intersection);
    }
}


std::set<std::string> CVolumeViewer::intersects()
{
    if (_segmentationOverlay) {
        return _segmentationOverlay->intersectsForViewer(*this);
    }
    return {};
}

void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    if (_segmentationOverlay) {
        _segmentationOverlay->setIntersectsForViewer(*this, set);
    }
}

void CVolumeViewer::fitSurfaceInView()
{
    if (!_surf || !fGraphicsView) {
        return;
    }

    Rect3D bbox;
    bool haveBounds = false;

    if (auto* quadSurf = dynamic_cast<QuadSurface*>(_surf)) {
        bbox = quadSurf->bbox();
        haveBounds = true;
    } else if (auto* opChain = dynamic_cast<OpChain*>(_surf)) {
        QuadSurface* src = opChain->src();
        if (src) {
            bbox = src->bbox();
            haveBounds = true;
        }
    }

    if (!haveBounds) {
        // when we can't get bounds, just reset to a default view
        _scale = 1.0f;
        recalcScales();
        fGraphicsView->resetTransform();
        fGraphicsView->centerOn(0, 0);
        _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
        return;
    }

    // Calculate the actual dimensions of the bounding box
    float bboxWidth = bbox.high[0] - bbox.low[0];
    float bboxHeight = bbox.high[1] - bbox.low[1];

    if (bboxWidth <= 0 || bboxHeight <= 0) {
        return;
    }

    QSize viewportSize = fGraphicsView->viewport()->size();
    float viewportWidth = viewportSize.width();
    float viewportHeight = viewportSize.height();

    if (viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    // Calculate scale factor based on actual bbox dimensions
    float fit_factor = 0.8f;
    float required_scale_x = (viewportWidth * fit_factor) / bboxWidth;
    float required_scale_y = (viewportHeight * fit_factor) / bboxHeight;

    // Use the smaller scale to ensure the entire bbox fits
    float required_scale = std::min(required_scale_x, required_scale_y);

    _scale = required_scale;
    round_scale(_scale);
    recalcScales();

    fGraphicsView->resetTransform();
    fGraphicsView->centerOn(0, 0);

    _lbl->setText(QString("%1x %2").arg(_scale).arg(_z_off));
    curr_img_area = {0,0,0,0};

    if (_bboxMode && isBBoxPlaneView()) {
        updateBBoxOverlay3D();
        QPointF viewportCenter = fGraphicsView->mapToScene(fGraphicsView->viewport()->rect().center());
        updateBBoxCursor(viewportCenter);
    }
}


QuadSurface* CVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surf_name != "segmentation") return nullptr;
    auto* quad = dynamic_cast<QuadSurface*>(_surf);
    if (!quad) return nullptr;

    const cv::Mat_<cv::Vec3f> src = quad->rawPoints();
    const int H = src.rows;
    const int W = src.cols;

    // Convert scene-space rect to surface-parameter rect (nominal units)
    QRectF rSurf(QPointF(sceneRect.left()/_scale,  sceneRect.top()/_scale),
                 QPointF(sceneRect.right()/_scale, sceneRect.bottom()/_scale));
    rSurf = rSurf.normalized();

    // Compute tight index bounds from surface-parameter rect
    const double cx = W * 0.5;
    const double cy = H * 0.5;
    const cv::Vec2f sc = quad->scale();
    int i0 = std::max(0,               static_cast<int>(std::floor(cx + rSurf.left()   * sc[0])));
    int i1 = std::min(W - 1,           static_cast<int>(std::ceil (cx + rSurf.right()  * sc[0])));
    int j0 = std::max(0,               static_cast<int>(std::floor(cy + rSurf.top()    * sc[1])));
    int j1 = std::min(H - 1,           static_cast<int>(std::ceil (cy + rSurf.bottom() * sc[1])));
    if (i0 > i1 || j0 > j1) return nullptr;

    const int outW = (i1 - i0 + 1);
    const int outH = (j1 - j0 + 1);
    cv::Mat_<cv::Vec3f> patch(outH, outW, cv::Vec3f(-1.f, -1.f, -1.f));
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            patch(j - j0, i - i0) = src(j, i);
        }
    }

    // Remove outliers to avoid straggling points
    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(patch, 5.0f);

    auto countValidInCol = [&](int c) {
        int cnt = 0; for (int r = 0; r < cleaned.rows; ++r) if (cleaned(r,c)[0] != -1) ++cnt; return cnt; };
    auto countValidInRow = [&](int r) {
        int cnt = 0; for (int c = 0; c < cleaned.cols; ++c) if (cleaned(r,c)[0] != -1) ++cnt; return cnt; };
    int minValidCol = std::max(1, std::min(3, cleaned.rows));
    int minValidRow = std::max(1, std::min(3, cleaned.cols));

    int left = 0, right = cleaned.cols - 1, top = 0, bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol) ++left;
    while (right >= left && countValidInCol(right) < minValidCol) --right;
    while (top <= bottom && countValidInRow(top) < minValidRow) ++top;
    while (bottom >= top && countValidInRow(bottom) < minValidRow) --bottom;

    if (left > right || top > bottom) {
        left = cleaned.cols; right = -1; top = cleaned.rows; bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j)
            for (int i = 0; i < cleaned.cols; ++i)
                if (cleaned(j,i)[0] != -1) {
                    left = std::min(left, i); right = std::max(right, i);
                    top  = std::min(top,  j); bottom= std::max(bottom,j);
                }
        if (right < 0 || bottom < 0) return nullptr;
    }

    const int fW = (right - left + 1);
    const int fH = (bottom - top + 1);
    cv::Mat_<cv::Vec3f> finalPts(fH, fW, cv::Vec3f(-1.f, -1.f, -1.f));
    for (int j = top; j <= bottom; ++j)
        for (int i = left; i <= right; ++i)
            finalPts(j - top, i - left) = cleaned(j, i);

    auto* result = new QuadSurface(finalPts, quad->_scale);
    return result;
}

auto CVolumeViewer::selections() const -> std::vector<std::pair<QRectF, QColor>>
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& s : _selections) {
        QRectF sceneRect(QPointF(s.surfRect.left()*_scale,  s.surfRect.top()*_scale),
                         QPointF(s.surfRect.right()*_scale, s.surfRect.bottom()*_scale));
        out.emplace_back(sceneRect.normalized(), s.color);
    }
    return out;
}

void CVolumeViewer::clearSelections()
{
    for (auto& s : _selections) {
        if (s.item) {
            fScene->removeItem(s.item);
            delete s.item;
        }
    }
    _selections.clear();
}

void CVolumeViewer::updateSelectionGraphics()
{
    for (auto& s : _selections) {
        if (!s.item) continue;
        QRectF sceneRect(QPointF(s.surfRect.left()*_scale,  s.surfRect.top()*_scale),
                         QPointF(s.surfRect.right()*_scale, s.surfRect.bottom()*_scale));
        s.item->setRect(sceneRect.normalized());
    }
}


void CVolumeViewer::onSurfaceChanged(std::string name, Surface *surf)
{
    if (_surf_name == name) {
        _surf = surf;
        if (!_surf) {
            if (_segmentationOverlay) {
                _segmentationOverlay->invalidateViewer(*this);
            }
            fScene->clear();
            slice_vis_items.clear();
            _points_items.clear();
            _path_items.clear();
            _paths.clear();
            // Scene items are already deleted by fScene->clear(); just drop overlay references
            _overlay_groups.clear();
            _cursor = nullptr;
            _center_marker = nullptr;
            fBaseImageItem = nullptr;
        }
        else {
            invalidateVis();
            _z_off = 0.0f;
            if (name == "segmentation" && _resetViewOnSurfaceChange) {
                fitSurfaceInView();
            }
        }

        if (_bboxMode) {
            if (isBBoxPlaneView()) {
                updateBBoxOverlay3D();
            } else {
                clearBBoxOverlay3D();
            }
        }
    }

    if (name == _surf_name) {
        curr_img_area = {0,0,0,0};
        renderVisible(true); // Immediate render of slice
    }

    // Defer overlay updates
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

QGraphicsItem *cursorItem(bool drawingMode = false, float brushSize = 3.0f, bool isSquare = false)
{
    if (drawingMode) {
        // Drawing mode cursor - shows brush shape and size
        QGraphicsItemGroup *group = new QGraphicsItemGroup();
        group->setZValue(10);
        
        QPen brushPen(QBrush(COLOR_CURSOR), 1.5);
        brushPen.setStyle(Qt::DashLine);
        
        // Draw brush shape
        if (isSquare) {
            float halfSize = brushSize / 2.0f;
            QGraphicsRectItem *rect = new QGraphicsRectItem(-halfSize, -halfSize, brushSize, brushSize);
            rect->setPen(brushPen);
            rect->setBrush(Qt::NoBrush);
            group->addToGroup(rect);
        } else {
            QGraphicsEllipseItem *circle = new QGraphicsEllipseItem(-brushSize/2, -brushSize/2, brushSize, brushSize);
            circle->setPen(brushPen);
            circle->setBrush(Qt::NoBrush);
            group->addToGroup(circle);
        }
        
        // Add small crosshair in center
        QPen centerPen(QBrush(COLOR_CURSOR), 1);
        QGraphicsLineItem *line = new QGraphicsLineItem(-2, 0, 2, 0);
        line->setPen(centerPen);
        group->addToGroup(line);
        line = new QGraphicsLineItem(0, -2, 0, 2);
        line->setPen(centerPen);
        group->addToGroup(line);
        
        return group;
    } else {
        // Regular cursor
        QPen pen(QBrush(COLOR_CURSOR), 2);
        QGraphicsLineItem *parent = new QGraphicsLineItem(-10, 0, -5, 0);
        parent->setZValue(10);
        parent->setPen(pen);
        QGraphicsLineItem *line = new QGraphicsLineItem(10, 0, 5, 0, parent);
        line->setPen(pen);
        line = new QGraphicsLineItem(0, -10, 0, -5, parent);
        line->setPen(pen);
        line = new QGraphicsLineItem(0, 10, 0, 5, parent);
        line->setPen(pen);
        
        return parent;
    }
}

QGraphicsItem *crossItem()
{
    QPen pen(QBrush(Qt::red), 1);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-5, -5, 5, 5);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(-5, 5, 5, -5, parent);
    line->setPen(pen);
    
    return parent;
}

//TODO make poi tracking optional and configurable
void CVolumeViewer::onPOIChanged(std::string name, POI *poi)
{    
    if (!poi || !_surf)
        return;
    
    if (name == "focus") {
        // Add safety check before dynamic_cast
        if (!_surf) {
            return;
        }
        
        if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
            fGraphicsView->centerOn(0,0);
            if (poi->p == plane->origin())
                return;
            
            plane->setOrigin(poi->p);
            refreshPointPositions();
            
            _surf_col->setSurface(_surf_name, plane);
        } else if (auto* quad = dynamic_cast<QuadSurface*>(_surf)) {
            auto ptr = quad->pointer();
            float dist = quad->pointTo(ptr, poi->p, 4.0, 100);
            
            if (dist < 4.0) {
                cv::Vec3f sp = quad->loc(ptr) * _scale;
                if (_center_marker) {
                    _center_marker->setPos(sp[0], sp[1]);
                    _center_marker->show();
                }
                fGraphicsView->centerOn(sp[0], sp[1]);
            } else {
                if (_center_marker) {
                    _center_marker->hide();
                }
            }
        }
    }
    else if (name == "cursor") {
        // Add safety check before dynamic_cast
        if (!_surf) {
            return;
        }
        
        PlaneSurface *slice_plane = dynamic_cast<PlaneSurface*>(_surf);
        // QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("visible_segmentation"));
        QuadSurface *crop = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
        
        cv::Vec3f sp;
        float dist = -1;
        if (slice_plane) {            
            dist = slice_plane->pointDist(poi->p);
            sp = slice_plane->project(poi->p, 1.0, _scale);
        }
        else if (_surf_name == "segmentation" && crop)
        {
            auto ptr = crop->pointer();
            dist = crop->pointTo(ptr, poi->p, 2.0);
            sp = crop->loc(ptr)*_scale ;//+ cv::Vec3f(_vis_center[0],_vis_center[1],0);
        }
        
        if (!_cursor) {
            _cursor = cursorItem(_drawingModeActive, _brushSize, _brushIsSquare);
            fScene->addItem(_cursor);
        }
        
        if (dist != -1) {
            if (dist < 20.0/_scale) {
                _cursor->setPos(sp[0], sp[1]);
                _cursor->setOpacity(1.0-dist*_scale/20.0);
            }
            else
                _cursor->setOpacity(0.0);
        }
    }
}

cv::Mat_<uint8_t> CVolumeViewer::render_composite(const cv::Rect &roi) {
    cv::Mat_<uint8_t> img;

    // Composite rendering for segmentation view
    cv::Mat_<float> accumulator;
    int count = 0;

    // Alpha composition state for each pixel
    cv::Mat_<float> alpha_accumulator;
    cv::Mat_<float> value_accumulator;

    // Alpha composition parameters using the new settings
    const float alpha_min = _composite_alpha_min / 255.0f;
    const float alpha_max = _composite_alpha_max / 255.0f;
    const float alpha_opacity = _composite_material / 255.0f;
    const float alpha_cutoff = _composite_alpha_threshold / 10000.0f;

    // Determine the z range based on front and behind layers
    int z_start = _composite_reverse_direction ? -_composite_layers_behind : -_composite_layers_front;
    int z_end = _composite_reverse_direction ? _composite_layers_front : _composite_layers_behind;

    for (int z = z_start; z <= z_end; z++) {
        cv::Mat_<cv::Vec3f> slice_coords;
        cv::Mat_<uint8_t> slice_img;

        cv::Vec2f roi_c = {roi.x+roi.width/2, roi.y + roi.height/2};
        _ptr = _surf->pointer();
        cv::Vec3f diff = {roi_c[0],roi_c[1],0};
        _surf->move(_ptr, diff/_scale);
        _vis_center = roi_c;
        float z_step = z * _ds_scale;  // Scale the step to maintain consistent physical distance
        _surf->gen(&slice_coords, nullptr, roi.size(), _ptr, _scale, {-roi.width/2, -roi.height/2, _z_off + z_step});

        readInterpolated3D(slice_img, volume->zarrDataset(_ds_sd_idx), slice_coords*_ds_scale, cache, _useFastInterpolation);

        // Convert to float for accumulation
        cv::Mat_<float> slice_float;
        slice_img.convertTo(slice_float, CV_32F);

        if (_composite_method == "alpha") {
            // Alpha composition algorithm
            if (alpha_accumulator.empty()) {
                alpha_accumulator = cv::Mat_<float>::zeros(slice_float.size());
                value_accumulator = cv::Mat_<float>::zeros(slice_float.size());
            }

            // Process each pixel
            for (int y = 0; y < slice_float.rows; y++) {
                for (int x = 0; x < slice_float.cols; x++) {
                    float pixel_value = slice_float(y, x);

                    // Normalize pixel value
                    float normalized_value = (pixel_value / 255.0f - alpha_min) / (alpha_max - alpha_min);
                    normalized_value = std::max(0.0f, std::min(1.0f, normalized_value)); // Clamp to [0,1]

                    // Skip empty areas (speed through)
                    if (normalized_value == 0.0f) {
                        continue;
                    }

                    float current_alpha = alpha_accumulator(y, x);

                    // Check alpha cutoff for early termination
                    if (current_alpha >= alpha_cutoff) {
                        continue;
                    }

                    // Calculate weight
                    float weight = (1.0f - current_alpha) * std::min(normalized_value * alpha_opacity, 1.0f);

                    // Accumulate
                    value_accumulator(y, x) += weight * normalized_value;
                    alpha_accumulator(y, x) += weight;
                }
            }
        } else {
            // Original composite methods
            if (accumulator.empty()) {
                accumulator = slice_float;
                if (_composite_method == "min") {
                    accumulator.setTo(255.0); // Initialize to max value for min operation
                    accumulator = cv::min(accumulator, slice_float);
                }
            } else {
                if (_composite_method == "max") {
                    accumulator = cv::max(accumulator, slice_float);
                } else if (_composite_method == "mean") {
                    accumulator += slice_float;
                    count++;
                } else if (_composite_method == "min") {
                    accumulator = cv::min(accumulator, slice_float);
                }
            }
        }
    }

    // Finalize alpha composition result
    if (_composite_method == "alpha") {
        accumulator = cv::Mat_<float>::zeros(value_accumulator.size());
        for (int y = 0; y < value_accumulator.rows; y++) {
            for (int x = 0; x < value_accumulator.cols; x++) {
                float final_value = value_accumulator(y, x) * 255.0f;
                accumulator(y, x) = std::max(0.0f, std::min(255.0f, final_value)); // Clamp to [0,255]
            }
        }
    }

    // Convert back to uint8
    if (_composite_method == "mean" && count > 0) {
        accumulator /= count;
    }
    accumulator.convertTo(img, CV_8U);
    return img;
}

cv::Mat_<uint8_t> CVolumeViewer::renderCompositeForSurface(QuadSurface* surface, cv::Size outputSize)
{
    if (!surface || !_composite_enabled || !volume) {
        return cv::Mat_<uint8_t>();
    }

    // Save current state
    float oldScale = _scale;
    cv::Vec2f oldVisCenter = _vis_center;
    Surface* oldSurf = _surf;
    float oldZOff = _z_off;
    cv::Vec3f oldPtr = _ptr;
    float oldDsScale = _ds_scale;
    int oldDsSdIdx = _ds_sd_idx;

    // Set up for surface rendering at 1:1 scale
    _surf = surface;
    _scale = 1.0f;
    _z_off = 0.0f;

    recalcScales();
    _ptr = _surf->pointer();
    cv::Rect roi(-outputSize.width/2, -outputSize.height/2,
                 outputSize.width, outputSize.height);

    _vis_center = cv::Vec2f(0, 0);

    cv::Mat_<uint8_t> result = render_composite(roi);

    _surf = oldSurf;
    _scale = oldScale;
    _vis_center = oldVisCenter;
    _z_off = oldZOff;
    _ptr = oldPtr;
    _ds_scale = oldDsScale;
    _ds_sd_idx = oldDsSdIdx;

    return result;
}


cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> img;

    // Check if we should use composite rendering
    if (_surf_name == "segmentation" && _composite_enabled && (_composite_layers_front > 0 || _composite_layers_behind > 0)) {
        img = render_composite(roi);
    }
    else {
        // Standard single-slice rendering
        //PlaneSurface use absolute positioning to simplify intersection logic
        if (dynamic_cast<PlaneSurface*>(_surf)) {
            _surf->gen(&coords, nullptr, roi.size(), cv::Vec3f(0,0,0), _scale, {roi.x, roi.y, _z_off});
        }
        else {
            cv::Vec2f roi_c = {roi.x+roi.width/2, roi.y + roi.height/2};

            _ptr = _surf->pointer();
            cv::Vec3f diff = {roi_c[0],roi_c[1],0};
            _surf->move(_ptr, diff/_scale);
            _vis_center = roi_c;
            _surf->gen(&coords, nullptr, roi.size(), _ptr, _scale, {-roi.width/2, -roi.height/2, _z_off});
        }

        readInterpolated3D(img, volume->zarrDataset(_ds_sd_idx), coords*_ds_scale, cache, _useFastInterpolation);
    }
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    return img;
}

class LifeTime
{
public:
    LifeTime(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~LifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

void CVolumeViewer::renderVisible(bool force)
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;
    
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
    if (!force && QRectF(curr_img_area).contains(bbox))
        return;
    
    renderPaths();
    
    curr_img_area = {bbox.left(),bbox.top(), bbox.width(), bbox.height()};
    
    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});
    
    QImage qimg = Mat2QImage(img);
    
    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);
 
    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);
    
    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
    }

    _center_marker->setParentItem(fBaseImageItem);
    
    fBaseImageItem->setOffset(curr_img_area.topLeft());
}



void CVolumeViewer::renderIntersections()
{
    if (_segmentationOverlay) {
        _segmentationOverlay->refreshViewer(*this);
    }
}


void CVolumeViewer::onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    renderVisible();

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CVolumeViewer::onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    renderVisible();

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CVolumeViewer::onScrolled()
{
    // if (!dynamic_cast<OpChain*>(_surf) && !dynamic_cast<OpChain*>(_surf)->slow() && _min_scale == 1.0)
        // renderVisible();
    // if ((!dynamic_cast<OpChain*>(_surf) || !dynamic_cast<OpChain*>(_surf)->slow()) && _min_scale < 1.0)
        // renderVisible();
}

void CVolumeViewer::onResized()
{
   renderVisible(true);
}

void CVolumeViewer::renderPaths()
{
   // Clear existing path items
    for(auto &item : _path_items) {
        if (item && item->scene() == fScene) {
            fScene->removeItem(item);
        }
        delete item;
    }
    _path_items.clear();
    
    if (!_surf) {
        return;
    }
    
    // Separate paths by type for proper rendering order
    QList<PathData> drawPaths;
    QList<PathData> eraserPaths;
    
    for (const auto& path : _paths) {
        if (path.isEraser) {
            eraserPaths.append(path);
        } else {
            drawPaths.append(path);
        }
    }
    
    // First render regular drawing paths
    for (const auto& path : drawPaths) {
        if (path.points.size() < 2) {
            continue;
        }
        
        QPainterPath painterPath;
        bool firstPoint = true;
        
        PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
        QuadSurface *quad = dynamic_cast<QuadSurface*>(_surf);
        
        for (const auto& wp : path.points) {
            cv::Vec3f p;
            
            if (plane) {
                if (plane->pointDist(wp) >= 4.0)
                    continue;
                p = plane->project(wp, 1.0, _scale);
            }
            else if (quad) {
                auto ptr = quad->pointer();
                float res = _surf->pointTo(ptr, wp, 4.0, 100);
                p = _surf->loc(ptr)*_scale;
                if (res >= 4.0)
                    continue;
            }
            else
                continue;
            
            if (firstPoint) {
                painterPath.moveTo(p[0], p[1]);
                firstPoint = false;
            } else {
                painterPath.lineTo(p[0], p[1]);
            }
        }
        
        // Create the path item with the specified color and properties
        QColor color = path.color;
        if (path.opacity < 1.0f) {
            color.setAlphaF(path.opacity);
        }
        
        QPen pen(color, path.lineWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        
        // Apply different brush shapes
        if (path.brushShape == PathData::BrushShape::SQUARE) {
            pen.setCapStyle(Qt::SquareCap);
            pen.setJoinStyle(Qt::MiterJoin);
        }
        
        auto item = fScene->addPath(painterPath, pen);
        item->setZValue(25); // Higher than intersections but lower than points
        _path_items.push_back(item);
    }
    
    // Then render eraser paths with a distinctive style
    // In the actual mask generation, these will subtract from the drawn areas
    for (const auto& path : eraserPaths) {
        if (path.points.size() < 2) {
            continue;
        }
        
        QPainterPath painterPath;
        bool firstPoint = true;
        
        PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
        QuadSurface *quad = dynamic_cast<QuadSurface*>(_surf);
        
        for (const auto& wp : path.points) {
            cv::Vec3f p;
            
            if (plane) {
                if (plane->pointDist(wp) >= 4.0)
                    continue;
                p = plane->project(wp, 1.0, _scale);
            }
            else if (quad) {
                auto ptr = quad->pointer();
                float res = _surf->pointTo(ptr, wp, 4.0, 100);
                p = _surf->loc(ptr)*_scale;
                if (res >= 4.0)
                    continue;
            }
            else
                continue;
            
            if (firstPoint) {
                painterPath.moveTo(p[0], p[1]);
                firstPoint = false;
            } else {
                painterPath.lineTo(p[0], p[1]);
            }
        }
        
        // Render eraser paths with a distinctive appearance
        // Using a dashed pattern to indicate eraser mode
        QPen pen(Qt::red, path.lineWidth, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setDashPattern(QVector<qreal>() << 4 << 4);
        
        if (path.opacity < 1.0f) {
            QColor eraserColor = pen.color();
            eraserColor.setAlphaF(path.opacity);
            pen.setColor(eraserColor);
        }
        
        auto item = fScene->addPath(painterPath, pen);
        item->setZValue(26); // Slightly higher than regular paths
        _path_items.push_back(item);
    }
}

void CVolumeViewer::renderOrUpdatePoint(const ColPoint& point)
{
    if (!_surf) return;

    float opacity = 1.0f;
    float z_dist = -1.0f;

    if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
        z_dist = std::abs(plane->pointDist(point.p));
    } else if (auto* quad = dynamic_cast<QuadSurface*>(_surf)) {
        auto ptr = quad->pointer();
        z_dist = quad->pointTo(ptr, point.p, 10.0, 100);
    }

    if (z_dist >= 0) {
        const float fade_threshold = 10.0f; // Fade over N units
        if (z_dist < fade_threshold) {
            opacity = 1.0f - (z_dist / fade_threshold);
        } else {
            opacity = 0.0f;
        }
    }

    QPointF scene_pos = volumeToScene(point.p);
    float radius = 5.0f; // pixels
    
    const auto& collections = _point_collection->getAllCollections();
    auto col_it = collections.find(point.collectionId);
    cv::Vec3f cv_color = (col_it != collections.end()) ? col_it->second.color : cv::Vec3f(1,0,0);
    QColor color(cv_color[0] * 255, cv_color[1] * 255, cv_color[2] * 255, 255);

    QColor border_color(255, 255, 255, 200);
    float border_width = 1.5f;

    if (point.id == _highlighted_point_id) {
        radius = 7.0f;
        border_color = Qt::yellow;
        border_width = 2.5f;
    }
 
    if (point.id == _selected_point_id) {
        border_color = QColor(255, 0, 255, 255); // Bright magenta for selection
        border_width = 2.5f;
        radius = 7.0f;
    }

    PointGraphics pg;
    bool exists = _points_items.count(point.id);
    if (exists) {
        pg = _points_items[point.id];
    }

    // Update circle
    if (exists) {
        pg.circle->setRect(scene_pos.x() - radius, scene_pos.y() - radius, radius * 2, radius * 2);
        pg.circle->setPen(QPen(border_color, border_width));
        pg.circle->setBrush(QBrush(color));
    } else {
        pg.circle = fScene->addEllipse(
            scene_pos.x() - radius, scene_pos.y() - radius, radius * 2, radius * 2,
            QPen(border_color, border_width), QBrush(color)
        );
        pg.circle->setZValue(10);
    }
    pg.circle->setOpacity(opacity);

    // Update or create text
    bool has_winding = !std::isnan(point.winding_annotation);
    if (exists) {
        pg.text->setPos(scene_pos.x() + radius, scene_pos.y() - radius);
        pg.text->setVisible(has_winding);
    } else {
        pg.text = new COutlinedTextItem();
        fScene->addItem(pg.text);
        pg.text->setZValue(11); // Above points
        pg.text->setDefaultTextColor(Qt::white);
        pg.text->setPos(scene_pos.x() + radius, scene_pos.y() - radius);
        pg.text->setVisible(has_winding);
    }
    pg.text->setOpacity(opacity);
    
    if (has_winding) {
        bool absolute = col_it != collections.end() ? col_it->second.metadata.absolute_winding_number : false;
        
        // Adaptive decimal formatting
        QString num_text = QString::number(point.winding_annotation, 'g');

        if (!absolute) {
            if (point.winding_annotation >= 0) {
                num_text.prepend("+");
            }
        }
        
        pg.text->setPlainText(num_text);

        // Fixed positioning
        pg.text->setPos(scene_pos.x() + radius, scene_pos.y() - radius);
    }

    if (!exists) {
        _points_items[point.id] = pg;
    }
}

void CVolumeViewer::onPathsChanged(const QList<PathData>& paths)
{
    _paths = paths;
    renderPaths();
}

void CVolumeViewer::onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (button == Qt::LeftButton) {
            // Convert to surface parameter coords (unscaled)
            cv::Vec3f p, n;
            if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) return;
            auto* quad = dynamic_cast<QuadSurface*>(_surf);
            if (!quad) return;
            auto ptr = quad->pointer();
            quad->pointTo(ptr, p, 2.0f, 100);
            cv::Vec3f sp = quad->loc(ptr); // unscaled surface coords
            _bboxStart = QPointF(sp[0], sp[1]);
            if (_bboxRectItem) {
                fScene->removeItem(_bboxRectItem);
                delete _bboxRectItem;
                _bboxRectItem = nullptr;
            }
            QRectF r(QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale), QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale));
            _bboxRectItem = fScene->addRect(r, QPen(QColor(255, 220, 0), 2, Qt::DashLine));
            _bboxRectItem->setZValue(100);
        }
        return; // consume in bbox mode
    }

    if (_bboxMode && isBBoxPlaneView()) {
        if (handleBBoxPlaneMousePress(scene_loc, button)) {
            return;
        }
    }

    if (!_point_collection || !_surf) return;

    if (button == Qt::LeftButton) {
        if (_highlighted_point_id != 0 && !modifiers.testFlag(Qt::ControlModifier)) {
            emit pointClicked(_highlighted_point_id);
            _dragged_point_id = _highlighted_point_id;
            // Do not return, allow forwarding for other widgets
        }
    } else if (button == Qt::RightButton) {
        if (_highlighted_point_id != 0) {
            _point_collection->removePoint(_highlighted_point_id);
        }
    }

    // Forward for drawing widgets
    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        sendMousePressVolume(p, n, button, modifiers);
    }
}

void CVolumeViewer::onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (_bboxRectItem && (buttons & Qt::LeftButton)) {
            cv::Vec3f p, n;
            if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) return;
            auto* quad = dynamic_cast<QuadSurface*>(_surf);
            if (!quad) return;
            auto ptr = quad->pointer();
            quad->pointTo(ptr, p, 2.0f, 100);
            cv::Vec3f sp = quad->loc(ptr); // unscaled
            QPointF cur(sp[0], sp[1]);
            QRectF r(QPointF(_bboxStart.x()*_scale, _bboxStart.y()*_scale), QPointF(cur.x()*_scale, cur.y()*_scale));
            _bboxRectItem->setRect(r.normalized());
        }
        return; // consume in bbox mode
    }

    if (_bboxMode && isBBoxPlaneView()) {
        if (handleBBoxPlaneMouseMove(scene_loc, buttons)) {
            return;
        }
        updateBBoxCursor(scene_loc);
    }

    onCursorMove(scene_loc); // Keep highlighting up to date

    if ((buttons & Qt::LeftButton) && _dragged_point_id != 0) {
        cv::Vec3f p, n;
        if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
            if (auto point_opt = _point_collection->getPoint(_dragged_point_id)) {
                ColPoint updated_point = *point_opt;
                updated_point.p = p;
                _point_collection->updatePoint(updated_point);
            }
        }
    } else {
        if (!_surf) {
            return;
        }
        
        cv::Vec3f p, n;
        if (!scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale))
            return;
        
        emit sendMouseMoveVolume(p, buttons, modifiers);
    }
}

void CVolumeViewer::onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    // BBox drawing consumes mouse events on segmentation view
    if (_bboxMode && _surf_name == "segmentation") {
        if (button == Qt::LeftButton && _bboxRectItem) {
            // Determine final rect in surface parameter coords
            QRectF rScene = _bboxRectItem->rect().normalized();
            QRectF rSurf(QPointF(rScene.left()/_scale, rScene.top()/_scale), QPointF(rScene.right()/_scale, rScene.bottom()/_scale));
            // Promote this rectangle into a persistent selection with unique color (stored unscaled)
            // Generate a distinct color using HSV cycling
            int idx = static_cast<int>(_selections.size());
            QColor col = QColor::fromHsv((idx * 53) % 360, 200, 255);
            // Create persistent item with current scale
            _bboxRectItem->setPen(QPen(col, 2, Qt::DashLine));
            _selections.push_back({rSurf, col, _bboxRectItem});
            _bboxRectItem = nullptr; // end active drag
        }
        return; // consume in bbox mode
    }

    if (_bboxMode && isBBoxPlaneView()) {
        if (handleBBoxPlaneMouseRelease(scene_loc, button)) {
            return;
        }
    }

    if (button == Qt::LeftButton && _dragged_point_id != 0) {
        _dragged_point_id = 0;
        // Re-run highlight logic
        onCursorMove(scene_loc);
    }

    // Forward for drawing widgets
    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        if (dynamic_cast<PlaneSurface*>(_surf))
            emit sendMouseReleaseVolume(p, button, modifiers);
        else if (_surf_name == "segmentation")
            emit sendMouseReleaseVolume(p, button, modifiers);
        else
            std::cout << "FIXME: onMouseRelease()" << std::endl;
    }
}

void CVolumeViewer::setBBoxCallbacks(std::function<void(const OrientedBBox&, bool)> onUpdate,
                                     std::function<std::optional<OrientedBBox>()> sharedRequest)
{
    _bboxSharedUpdate = std::move(onUpdate);
    _bboxSharedRequest = std::move(sharedRequest);
}

void CVolumeViewer::setExternalBBox(const std::optional<OrientedBBox>& bbox)
{
    if (bbox && bbox->valid()) {
        _bboxSharedBox = *bbox;
    } else if (bbox) {
        _bboxSharedBox.reset();
    } else {
        _bboxSharedBox.reset();
    }

    if (_bboxMode && isBBoxPlaneView()) {
        updateBBoxOverlay3D();
    } else if (!_bboxMode) {
        clearBBoxOverlay3D();
    }
}

void CVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;

    if (!_bboxMode) {
        if (_bboxRectItem) {
            fScene->removeItem(_bboxRectItem);
            delete _bboxRectItem;
            _bboxRectItem = nullptr;
        }
        _bboxDragMode3D = BBoxDragMode::None;
        clearBBoxOverlay3D();
        if (fGraphicsView && fGraphicsView->viewport()) {
            fGraphicsView->viewport()->unsetCursor();
        }
        return;
    }

    if (_surf_name == "segmentation") {
        return;
    }

    if (!_bboxSharedBox && _bboxSharedRequest) {
        setExternalBBox(_bboxSharedRequest());
    } else if (isBBoxPlaneView()) {
        updateBBoxOverlay3D();
    }

    if (_bboxMode && isBBoxPlaneView() && fGraphicsView) {
        QPointF scenePoint = fGraphicsView->mapToScene(fGraphicsView->mapFromGlobal(QCursor::pos()));
        updateBBoxCursor(scenePoint);
    }
}

cv::Vec3f CVolumeViewer::planeUnitVector(int axis) const
{
    return axisIndexToUnit(axis);
}

bool CVolumeViewer::projectSceneToVolume(const QPointF& scene_loc, cv::Vec3f& out) const
{
    cv::Vec3f normal;
    if (!scene2vol(out, normal, _surf, _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        return false;
    }
    return true;
}

OrientedBBox CVolumeViewer::defaultBBoxFromPoints(const cv::Vec3f& a,
                                                  const cv::Vec3f& b,
                                                  const BBoxAxes& axes) const
{
    OrientedBBox box;
    box.axisU = planeUnitVector(axes.axisPrimary);
    box.axisV = planeUnitVector(axes.axisSecondary);
    box.axisN = planeUnitVector(axes.axisNormal);

    auto coordAlong = [&](const cv::Vec3f& p, const cv::Vec3f& axis) {
        return p.dot(axis);
    };

    float minU = std::min(coordAlong(a, box.axisU), coordAlong(b, box.axisU));
    float maxU = std::max(coordAlong(a, box.axisU), coordAlong(b, box.axisU));
    float minV = std::min(coordAlong(a, box.axisV), coordAlong(b, box.axisV));
    float maxV = std::max(coordAlong(a, box.axisV), coordAlong(b, box.axisV));
    float minN = std::min(coordAlong(a, box.axisN), coordAlong(b, box.axisN));
    float maxN = std::max(coordAlong(a, box.axisN), coordAlong(b, box.axisN));

    float centerU = (minU + maxU) * 0.5f;
    float centerV = (minV + maxV) * 0.5f;
    float centerN = (minN + maxN) * 0.5f;

    box.halfExtents = {
        std::max(0.5f * (maxU - minU), 0.5f),
        std::max(0.5f * (maxV - minV), 0.5f),
        std::max(0.5f * (maxN - minN), 0.0f)
    };

    box.center = box.axisU * centerU + box.axisV * centerV + box.axisN * centerN;

    std::cout << "[BBox] defaultBBoxFromPoints a=" << vecToString(a)
              << " b=" << vecToString(b)
              << " axisPrimary=" << axes.axisPrimary
              << " axisSecondary=" << axes.axisSecondary
              << " axisNormal=" << axes.axisNormal
              << " center=" << vecToString(box.center)
              << " halfExtents=" << vecToString(box.halfExtents)
              << " axisU=" << vecToString(box.axisU)
              << " axisV=" << vecToString(box.axisV)
              << " axisN=" << vecToString(box.axisN)
              << std::endl;

    return box;
}

void CVolumeViewer::ensureBBoxHandles()
{
    if (!_bbox3DPolygon) {
        _bbox3DPolygon = fScene->addPolygon(QPolygonF{}, QPen(QColor(255, 220, 0), 2, Qt::DashLine), QBrush(Qt::NoBrush));
        _bbox3DPolygon->setZValue(150);
    }

    for (int i = 0; i < 4; ++i) {
        if (!_bbox3DHandles[i]) {
            _bbox3DHandles[i] = fScene->addEllipse(0, 0, BBOX_HANDLE_DIAMETER, BBOX_HANDLE_DIAMETER,
                                                   QPen(QColor(255, 220, 0)), QBrush(QColor(255, 220, 0)));
            _bbox3DHandles[i]->setZValue(160);
        }
    }

    if (!_bboxRotationHandle) {
        _bboxRotationHandle = fScene->addEllipse(0, 0, BBOX_ROTATION_HANDLE_DIAMETER, BBOX_ROTATION_HANDLE_DIAMETER,
                                                 QPen(QColor(255, 220, 0)), QBrush(QColor(255, 220, 0)));
        _bboxRotationHandle->setZValue(155);
    }
}

void CVolumeViewer::clearBBoxOverlay3D()
{
    if (_bbox3DPolygon) {
        fScene->removeItem(_bbox3DPolygon);
        delete _bbox3DPolygon;
        _bbox3DPolygon = nullptr;
    }
    for (auto& handle : _bbox3DHandles) {
        if (handle) {
            fScene->removeItem(handle);
            delete handle;
            handle = nullptr;
        }
    }
    for (auto& center : _bbox3DHandleCenters) {
        center = QPointF();
    }
    if (_bboxRotationHandle) {
        fScene->removeItem(_bboxRotationHandle);
        delete _bboxRotationHandle;
        _bboxRotationHandle = nullptr;
    }

    _bboxHandleAxisOrder = {0, 1};

    if (fGraphicsView && fGraphicsView->viewport()) {
        QPoint viewPos = fGraphicsView->mapFromGlobal(QCursor::pos());
        QPointF scenePos = fGraphicsView->mapToScene(viewPos);
        updateBBoxCursor(scenePos);
    }
}

std::array<QPointF,4> CVolumeViewer::bboxSceneCorners(const OrientedBBox& box,
                                                     const cv::Vec3f& axisA, float extentA,
                                                     const cv::Vec3f& axisB, float extentB) const
{
    std::array<QPointF,4> out{};
    const cv::Vec3f axisNormA = normalizedAxis(axisA);
    const cv::Vec3f axisNormB = normalizedAxis(axisB);
    const cv::Vec3f offsets[4] = {
        -axisNormA * extentA - axisNormB * extentB,
        -axisNormA * extentA + axisNormB * extentB,
        axisNormA * extentA + axisNormB * extentB,
        axisNormA * extentA - axisNormB * extentB
    };

    for (int i = 0; i < 4; ++i) {
        const cv::Vec3f volPoint = box.center + offsets[i];
        out[i] = const_cast<CVolumeViewer*>(this)->volumeToScene(volPoint);
    }
    return out;
}

void CVolumeViewer::refreshRotationHandle(const std::array<QPointF,4>& cornersScene,
                                          const OrientedBBox& box,
                                          int secondaryAxisIndex)
{
    if (!_bboxRotationHandle) {
        return;
    }

    QPointF topMid = (cornersScene[1] + cornersScene[2]) * 0.5;
    const cv::Vec3f axisSecondary = normalizedAxis(bboxAxisByIndex(box, secondaryAxisIndex));
    const float extentSecondary = bboxHalfExtentByIndex(box, secondaryAxisIndex);
    cv::Vec3f topCenterVol = box.center + axisSecondary * extentSecondary;
    cv::Vec3f dirVol = box.center + axisSecondary * (extentSecondary + BBOX_ROTATION_HANDLE_OFFSET / std::max(1.0f, _scale));

    QPointF topDirScene = volumeToScene(dirVol);
    QPointF direction = topDirScene - topMid;
    if (QLineF(QPointF(0,0), direction).length() < 1e-3) {
        direction = QPointF(0, -BBOX_ROTATION_HANDLE_OFFSET);
    }
    QLineF line(QPointF(0,0), direction);
    line.setLength(BBOX_ROTATION_HANDLE_OFFSET);

    QPointF handleCenter = topMid + line.p2();
    _bboxRotationHandleCenter = handleCenter;
    _bboxRotationHandle->setRect(handleCenter.x() - BBOX_ROTATION_HANDLE_RADIUS,
                                 handleCenter.y() - BBOX_ROTATION_HANDLE_RADIUS,
                                 BBOX_ROTATION_HANDLE_DIAMETER,
                                 BBOX_ROTATION_HANDLE_DIAMETER);
    _bboxRotationHandle->setVisible(true);
}

bool CVolumeViewer::bboxIntersectsCurrentPlane(const OrientedBBox& box) const
{
    auto* plane = dynamic_cast<PlaneSurface*>(_surf);
    if (!plane) {
        return true;
    }

    cv::Vec3f planeNormal = plane->normal(cv::Vec3f{}, cv::Vec3f{});
    const float normalLength = cv::norm(planeNormal);
    if (normalLength <= 1e-6f) {
        return true;
    }
    planeNormal /= normalLength;

    const cv::Vec3f planeOrigin = plane->origin();
    const float signedDistance = (box.center - planeOrigin).dot(planeNormal);
    const float support = bboxSupportAlongNormal(box, planeNormal);
    constexpr float kPlaneTolerance = 0.5f;

    return std::abs(signedDistance) <= (support + kPlaneTolerance);
}

void CVolumeViewer::syncOverlayToBox(const OrientedBBox& box)
{
    ensureBBoxHandles();

    auto axesOpt = bboxAxes();
    std::array<int,2> axisOrder{0, 1};
    cv::Vec3f axisA = box.axisU;
    cv::Vec3f axisB = box.axisV;
    float extentA = box.halfExtents[0];
    float extentB = box.halfExtents[1];

    if (axesOpt) {
        if (auto mappingOpt = mapDatasetAxesToBoxAxes(box)) {
            const auto& mapping = *mappingOpt;
            int idxPrimary = mapping[axesOpt->axisPrimary];
            int idxSecondary = mapping[axesOpt->axisSecondary];
            if (idxPrimary >= 0 && idxSecondary >= 0) {
                axisA = bboxAxisByIndex(box, idxPrimary);
                axisB = bboxAxisByIndex(box, idxSecondary);
                extentA = bboxHalfExtentByIndex(box, idxPrimary);
                extentB = bboxHalfExtentByIndex(box, idxSecondary);
                axisOrder = {idxPrimary, idxSecondary};
            }
        }
    }

    _bboxHandleAxisOrder = axisOrder;

    auto cornersScene = bboxSceneCorners(box, axisA, extentA, axisB, extentB);

    QPolygonF poly;
    for (const auto& pt : cornersScene) {
        poly << pt;
    }

    if (_bbox3DPolygon) {
        _bbox3DPolygon->setPolygon(poly);
        _bbox3DPolygon->setVisible(true);
    }

    const QPointF handleCenters[4] = {
        (cornersScene[0] + cornersScene[1]) * 0.5,
        (cornersScene[2] + cornersScene[3]) * 0.5,
        (cornersScene[0] + cornersScene[3]) * 0.5,
        (cornersScene[1] + cornersScene[2]) * 0.5
    };

    for (int i = 0; i < 4; ++i) {
        _bbox3DHandleCenters[i] = handleCenters[i];
        auto* handle = _bbox3DHandles[i];
        handle->setRect(handleCenters[i].x() - BBOX_HANDLE_RADIUS,
                        handleCenters[i].y() - BBOX_HANDLE_RADIUS,
                        BBOX_HANDLE_DIAMETER,
                        BBOX_HANDLE_DIAMETER);
        handle->setVisible(true);
    }

    refreshRotationHandle(cornersScene, box, _bboxHandleAxisOrder[1]);
}

bool CVolumeViewer::pickRotationHandle(const QPointF& scene_loc) const
{
    if (!_bboxRotationHandle || !_bboxRotationHandle->isVisible()) {
        return false;
    }
    QPointF diff = scene_loc - _bboxRotationHandleCenter;
    return QPointF::dotProduct(diff, diff) <= BBOX_HANDLE_PICK_RADIUS * BBOX_HANDLE_PICK_RADIUS;
}

bool CVolumeViewer::clampBBoxToVolume(OrientedBBox& box) const
{
    if (!volume) {
        return false;
    }
    // For now, rely on downstream sampling to clamp; return true to indicate no adjustment.
    (void)box;
    return true;
}

std::optional<CVolumeViewer::BBoxAxes> CVolumeViewer::bboxAxes() const
{
    BBoxAxes axes;
    if (_surf_name == "xy plane" || _surf_name == "xy" || _surf_name == "xy slices") {
        axes = {0, 1, 2};
    } else if (_surf_name == "seg xz" || _surf_name == "xz plane" || _surf_name == "segmentation xz") {
        axes = {0, 2, 1};
    } else if (_surf_name == "seg yz" || _surf_name == "yz plane" || _surf_name == "segmentation yz") {
        axes = {1, 2, 0};
    } else {
        return std::nullopt;
    }
    return axes;
}

bool CVolumeViewer::isBBoxPlaneView() const
{
    return bboxAxes().has_value() && dynamic_cast<PlaneSurface*>(_surf);
}

void CVolumeViewer::updateBBoxOverlay3D()
{
    if (!_bboxMode || !isBBoxPlaneView()) {
        clearBBoxOverlay3D();
        return;
    }

    if (!_bboxSharedBox || !_bboxSharedBox->valid()) {
        clearBBoxOverlay3D();
        return;
    }

    if (!bboxIntersectsCurrentPlane(*_bboxSharedBox)) {
        clearBBoxOverlay3D();
        return;
    }

    syncOverlayToBox(*_bboxSharedBox);

    if (fGraphicsView && fGraphicsView->viewport()) {
        QPoint viewPos = fGraphicsView->mapFromGlobal(QCursor::pos());
        QPointF scenePos = fGraphicsView->mapToScene(viewPos);
        updateBBoxCursor(scenePos);
    }
}

bool CVolumeViewer::handleBBoxPlaneMousePress(QPointF scene_loc, Qt::MouseButton button)
{
    if (button != Qt::LeftButton || !_bboxMode || !isBBoxPlaneView()) {
        return false;
    }

    if (!_bboxSharedBox && _bboxSharedRequest) {
        _bboxSharedBox = _bboxSharedRequest();
    }

    cv::Vec3f volumePoint;
    if (!projectSceneToVolume(scene_loc, volumePoint)) {
        return false;
    }

    if (_bboxSharedBox && pickRotationHandle(scene_loc)) {
        _bboxDragMode3D = BBoxDragMode::Rotate;
        _bboxDragInitialBox = *_bboxSharedBox;
        _bboxRotationAngleValid = false;
        int idxPrimary = 0;
        int idxSecondary = 1;
        int idxNormal = 2;
        if (auto axesOptRotation = bboxAxes()) {
            std::array<int, 3> datasetToBox{{0, 1, 2}};
            if (auto mappingOpt = mapDatasetAxesToBoxAxes(_bboxDragInitialBox)) {
                datasetToBox = *mappingOpt;
            }
            auto resolveAxisIndex = [&](int datasetAxis, int fallback) {
                if (datasetAxis < 0 || datasetAxis >= 3) {
                    return fallback;
                }
                int idx = datasetToBox[datasetAxis];
                if (idx < 0 || idx > 2) {
                    return fallback;
                }
                return idx;
            };
            idxPrimary = _bboxHandleAxisOrder[0];
            if (idxPrimary < 0 || idxPrimary > 2) {
                idxPrimary = resolveAxisIndex(axesOptRotation->axisPrimary, 0);
            }
            idxSecondary = _bboxHandleAxisOrder[1];
            if (idxSecondary < 0 || idxSecondary > 2) {
                idxSecondary = resolveAxisIndex(axesOptRotation->axisSecondary, 1);
            }
            idxNormal = resolveAxisIndex(axesOptRotation->axisNormal, 2);
        }

        cv::Vec3f axisPrimaryVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxPrimary));
        cv::Vec3f axisSecondaryVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxSecondary));
        cv::Vec3f axisNormalVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxNormal));
        if (auto angleOpt = rotationAngleForPoint(_bboxDragInitialBox.center,
                                                  volumePoint,
                                                  axisPrimaryVec,
                                                  axisSecondaryVec,
                                                  axisNormalVec)) {
            _bboxRotationStartAngle = *angleOpt;
            _bboxRotationAngleValid = true;
        }
        if (fGraphicsView && fGraphicsView->viewport()) {
            fGraphicsView->viewport()->setCursor(Qt::ClosedHandCursor);
        }
        return true;
    }

    int handleIdx = -1;
    for (int i = 0; i < 4; ++i) {
        if (!_bbox3DHandles[i] || !_bbox3DHandles[i]->isVisible()) {
            continue;
        }
        QPointF diff = scene_loc - _bbox3DHandleCenters[i];
        if (QPointF::dotProduct(diff, diff) <= BBOX_HANDLE_PICK_RADIUS * BBOX_HANDLE_PICK_RADIUS) {
            handleIdx = i;
            break;
        }
    }

    if (handleIdx >= 0) {
        _bboxDragInitialBox = _bboxSharedBox ? *_bboxSharedBox : OrientedBBox{};
        switch (handleIdx) {
        case 0: _bboxDragMode3D = BBoxDragMode::Axis0Min; break;
        case 1: _bboxDragMode3D = BBoxDragMode::Axis0Max; break;
        case 2: _bboxDragMode3D = BBoxDragMode::Axis1Min; break;
        case 3: _bboxDragMode3D = BBoxDragMode::Axis1Max; break;
        default: _bboxDragMode3D = BBoxDragMode::None; break;
        }
        return true;
    }

    if (_bbox3DPolygon && _bbox3DPolygon->isVisible() && _bboxSharedBox) {
        QPointF localPoint = _bbox3DPolygon->mapFromScene(scene_loc);
        if (_bbox3DPolygon->polygon().containsPoint(localPoint, Qt::OddEvenFill)) {
            _bboxDragMode3D = BBoxDragMode::Translate;
            _bboxDragStartVol = volumePoint;
            _bboxDragInitialBox = *_bboxSharedBox;
            if (fGraphicsView && fGraphicsView->viewport()) {
                fGraphicsView->viewport()->setCursor(Qt::ClosedHandCursor);
            }
            return true;
        }
    }

    auto axesOpt = bboxAxes();
    if (!axesOpt) {
        return false;
    }

    _bboxDragMode3D = BBoxDragMode::Create;
    _bboxDragStartVol = volumePoint;
    _bboxDragInitialBox = OrientedBBox{};
    OrientedBBox box = defaultBBoxFromPoints(volumePoint, volumePoint, *axesOpt);
    _bboxSharedBox = box;
    updateBBoxOverlay3D();
    return true;
}

bool CVolumeViewer::handleBBoxPlaneMouseMove(QPointF scene_loc, Qt::MouseButtons buttons)
{
    if (!_bboxMode || !isBBoxPlaneView()) {
        return false;
    }
    if (_bboxDragMode3D == BBoxDragMode::None) {
        updateBBoxCursor(scene_loc);
        return false;
    }
    if (!(buttons & Qt::LeftButton)) {
        return true;
    }

    cv::Vec3f volumePoint;
    if (!projectSceneToVolume(scene_loc, volumePoint)) {
        return true;
    }

    if (!_bboxSharedBox) {
        auto axesOpt = bboxAxes();
        if (!axesOpt) {
            return true;
        }
        _bboxSharedBox = defaultBBoxFromPoints(_bboxDragStartVol, volumePoint, *axesOpt);
    }

    OrientedBBox box = *_bboxSharedBox;
    auto axesOpt = bboxAxes();
    if (!axesOpt) {
        return true;
    }

    const auto& axes = *axesOpt;

    std::array<int,3> datasetToBox{{0, 1, 2}};
    if (auto mappingOpt = mapDatasetAxesToBoxAxes(_bboxDragInitialBox)) {
        datasetToBox = *mappingOpt;
    }

    auto resolveAxisIndex = [&](int datasetAxis, int fallback) {
        if (datasetAxis < 0 || datasetAxis >= 3) {
            return fallback;
        }
        int idx = datasetToBox[datasetAxis];
        if (idx < 0 || idx > 2) {
            return fallback;
        }
        return idx;
    };

    int idxPrimary = _bboxHandleAxisOrder[0];
    if (idxPrimary < 0 || idxPrimary > 2) {
        idxPrimary = resolveAxisIndex(axes.axisPrimary, 0);
    }
    int idxSecondary = _bboxHandleAxisOrder[1];
    if (idxSecondary < 0 || idxSecondary > 2) {
        idxSecondary = resolveAxisIndex(axes.axisSecondary, 1);
    }
    int idxNormal = resolveAxisIndex(axes.axisNormal, 2);

    const cv::Vec3f axisPrimaryVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxPrimary));
    const cv::Vec3f axisSecondaryVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxSecondary));
    const cv::Vec3f axisNormalVec = normalizedAxis(bboxAxisByIndex(_bboxDragInitialBox, idxNormal));

    const float initialExtentPrimary = bboxHalfExtentByIndex(_bboxDragInitialBox, idxPrimary);
    const float initialExtentSecondary = bboxHalfExtentByIndex(_bboxDragInitialBox, idxSecondary);

    auto coordAlong = [](const cv::Vec3f& p, const cv::Vec3f& axis) {
        return p.dot(axis);
    };

    switch (_bboxDragMode3D) {
    case BBoxDragMode::Axis0Min: {
        float right = initialExtentPrimary;
        float value = coordAlong(volumePoint - _bboxDragInitialBox.center, axisPrimaryVec);
        value = std::min(value, right - 0.5f);
        float left = std::min(value, right - 0.5f);
        float extent = (right - left) * 0.5f;
        float centerShift = (right + left) * 0.5f;
        bboxHalfExtentRefByIndex(box, idxPrimary) = std::max(extent, 0.5f);
        box.center = _bboxDragInitialBox.center + axisPrimaryVec * centerShift;
        break;
    }
    case BBoxDragMode::Axis0Max: {
        float left = -initialExtentPrimary;
        float value = coordAlong(volumePoint - _bboxDragInitialBox.center, axisPrimaryVec);
        value = std::max(value, left + 0.5f);
        float right = std::max(value, left + 0.5f);
        float extent = (right - left) * 0.5f;
        float centerShift = (right + left) * 0.5f;
        bboxHalfExtentRefByIndex(box, idxPrimary) = std::max(extent, 0.5f);
        box.center = _bboxDragInitialBox.center + axisPrimaryVec * centerShift;
        break;
    }
    case BBoxDragMode::Axis1Min: {
        float top = initialExtentSecondary;
        float value = coordAlong(volumePoint - _bboxDragInitialBox.center, axisSecondaryVec);
        value = std::min(value, top - 0.5f);
        float bottom = std::min(value, top - 0.5f);
        float extent = (top - bottom) * 0.5f;
        float centerShift = (top + bottom) * 0.5f;
        bboxHalfExtentRefByIndex(box, idxSecondary) = std::max(extent, 0.5f);
        box.center = _bboxDragInitialBox.center + axisSecondaryVec * centerShift;
        break;
    }
    case BBoxDragMode::Axis1Max: {
        float bottom = -initialExtentSecondary;
        float value = coordAlong(volumePoint - _bboxDragInitialBox.center, axisSecondaryVec);
        value = std::max(value, bottom + 0.5f);
        float top = std::max(value, bottom + 0.5f);
        float extent = (top - bottom) * 0.5f;
        float centerShift = (top + bottom) * 0.5f;
        bboxHalfExtentRefByIndex(box, idxSecondary) = std::max(extent, 0.5f);
        box.center = _bboxDragInitialBox.center + axisSecondaryVec * centerShift;
        break;
    }
    case BBoxDragMode::Translate: {
        cv::Vec3f delta = volumePoint - _bboxDragStartVol;
        float du = coordAlong(delta, axisPrimaryVec);
        float dv = coordAlong(delta, axisSecondaryVec);
        box = _bboxDragInitialBox;
        box.center = _bboxDragInitialBox.center + axisPrimaryVec * du + axisSecondaryVec * dv;
        break;
    }
    case BBoxDragMode::Create: {
        box = defaultBBoxFromPoints(_bboxDragStartVol, volumePoint, axes);
        break;
    }
    case BBoxDragMode::Rotate: {
        if (!_bboxRotationAngleValid) {
            break;
        }
        auto angleOpt = rotationAngleForPoint(_bboxDragInitialBox.center,
                                              volumePoint,
                                              axisPrimaryVec,
                                              axisSecondaryVec,
                                              axisNormalVec);
        if (!angleOpt) {
            break;
        }
        float delta = normalizeAngle(*angleOpt - _bboxRotationStartAngle);
        float c = std::cos(delta);
        float s = std::sin(delta);
        cv::Vec3f newPrimary = normalizedAxis(axisPrimaryVec * c + axisSecondaryVec * s);
        cv::Vec3f newSecondary = normalizedAxis(axisSecondaryVec * c - axisPrimaryVec * s);
        setBBoxAxisByIndex(box, idxPrimary, newPrimary);
        setBBoxAxisByIndex(box, idxSecondary, newSecondary);
        setBBoxAxisByIndex(box, idxNormal, axisNormalVec);
        box.center = _bboxDragInitialBox.center;
        box.halfExtents = _bboxDragInitialBox.halfExtents;
        break;
    }
    default:
        break;
    }

    clampBBoxToVolume(box);
    _bboxSharedBox = box;
    updateBBoxOverlay3D();
    if (_bboxSharedUpdate) {
        _bboxSharedUpdate(box, false);
    }
    return true;
}

bool CVolumeViewer::handleBBoxPlaneMouseRelease(QPointF scene_loc, Qt::MouseButton button)
{
    if (button != Qt::LeftButton || !_bboxMode || !isBBoxPlaneView()) {
        return false;
    }
    if (_bboxDragMode3D == BBoxDragMode::None) {
        return false;
    }

    handleBBoxPlaneMouseMove(scene_loc, Qt::MouseButtons(Qt::LeftButton));

    if (_bboxSharedBox && _bboxSharedUpdate) {
        _bboxSharedUpdate(*_bboxSharedBox, true);
    }
    _bboxDragMode3D = BBoxDragMode::None;
    _bboxRotationAngleValid = false;
    if (fGraphicsView && fGraphicsView->viewport()) {
        fGraphicsView->viewport()->unsetCursor();
        updateBBoxCursor(scene_loc);
    }
    return true;
}

void CVolumeViewer::updateBBoxCursor(const QPointF& scene_loc)
{
    if (!fGraphicsView || !fGraphicsView->viewport()) {
        return;
    }

    QWidget* viewport = fGraphicsView->viewport();

    if (!_bboxMode || !isBBoxPlaneView()) {
        viewport->unsetCursor();
        return;
    }

    if (_bboxDragMode3D != BBoxDragMode::None) {
        return;
    }

    auto setCursorShape = [&](Qt::CursorShape shape) {
        if (viewport->cursor().shape() != shape) {
            viewport->setCursor(shape);
        }
    };

    if (pickRotationHandle(scene_loc)) {
        setCursorShape(Qt::CrossCursor);
        return;
    }

    const double pickRadiusSq = BBOX_HANDLE_PICK_RADIUS * BBOX_HANDLE_PICK_RADIUS;
    for (int i = 0; i < 4; ++i) {
        auto* handle = _bbox3DHandles[i];
        if (!handle || !handle->isVisible()) {
            continue;
        }
        QPointF diff = scene_loc - _bbox3DHandleCenters[i];
        if (QPointF::dotProduct(diff, diff) <= pickRadiusSq) {
            setCursorShape(Qt::OpenHandCursor);
            return;
        }
    }

    bool insidePolygon = false;
    if (_bbox3DPolygon && _bbox3DPolygon->isVisible()) {
        QPointF localPoint = _bbox3DPolygon->mapFromScene(scene_loc);
        insidePolygon = _bbox3DPolygon->polygon().containsPoint(localPoint, Qt::OddEvenFill);
    }

    if (insidePolygon) {
        setCursorShape(Qt::OpenHandCursor);
        return;
    }

    viewport->unsetCursor();
}

void CVolumeViewer::setCompositeEnabled(bool enabled)
{
    if (_composite_enabled != enabled) {
        _composite_enabled = enabled;
        renderVisible(true);
        
        // Update status label
        QString status = QString("%1x %2").arg(_scale).arg(_z_off);
        if (_composite_enabled) {
            QString method = QString::fromStdString(_composite_method);
            method[0] = method[0].toUpper();
            status += QString(" | Composite: %1(%2)").arg(method).arg(_composite_layers);
        }
        _lbl->setText(status);
    }
}

void CVolumeViewer::setCompositeLayers(int layers)
{
    if (layers >= 1 && layers <= 21 && layers != _composite_layers) {
        _composite_layers = layers;
        if (_composite_enabled) {
            renderVisible(true);
            
            // Update status label
            QString status = QString("%1x %2").arg(_scale).arg(_z_off);
            QString method = QString::fromStdString(_composite_method);
            method[0] = method[0].toUpper();
            status += QString(" | Composite: %1(%2)").arg(method).arg(_composite_layers);
            _lbl->setText(status);
        }
    }
}

void CVolumeViewer::setCompositeLayersInFront(int layers)
{
    if (layers >= 0 && layers <= 21 && layers != _composite_layers_front) {
        _composite_layers_front = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeLayersBehind(int layers)
{
    if (layers >= 0 && layers <= 21 && layers != _composite_layers_behind) {
        _composite_layers_behind = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMin(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_min) {
        _composite_alpha_min = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMax(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_max) {
        _composite_alpha_max = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaThreshold(int value)
{
    if (value >= 0 && value <= 10000 && value != _composite_alpha_threshold) {
        _composite_alpha_threshold = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMaterial(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_material) {
        _composite_material = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeReverseDirection(bool reverse)
{
    if (reverse != _composite_reverse_direction) {
        _composite_reverse_direction = reverse;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMethod(const std::string& method)
{
    if (method != _composite_method && (method == "max" || method == "mean" || method == "min" || method == "alpha")) {
        _composite_method = method;
        if (_composite_enabled) {
            renderVisible(true);
            
            // Update status label
            QString status = QString("%1x %2").arg(_scale).arg(_z_off);
            QString methodDisplay = QString::fromStdString(_composite_method);
            methodDisplay[0] = methodDisplay[0].toUpper();
            status += QString(" | Composite: %1(%2)").arg(methodDisplay).arg(_composite_layers);
            _lbl->setText(status);
        }
    }
}

void CVolumeViewer::onVolumeClosing()
{
    // Only clear segmentation-related surfaces, not persistent plane surfaces
    if (_surf_name == "segmentation") {
        onSurfaceChanged(_surf_name, nullptr);
    }
    // For plane surfaces (xy plane, xz plane, yz plane), just clear the scene
    // but keep the surface reference so it can render with the new volume
    else if (_surf_name == "xy plane" || _surf_name == "xz plane" || _surf_name == "yz plane") {
        if (_segmentationOverlay) {
            _segmentationOverlay->invalidateViewer(*this);
        }
        if (fScene) {
            fScene->clear();
        }
        // Clear all item collections
        slice_vis_items.clear();
        _points_items.clear();
        _path_items.clear();
        _paths.clear();
        _cursor = nullptr;
        _center_marker = nullptr;
        fBaseImageItem = nullptr;
        // Note: We don't set _surf = nullptr here, so the surface remains available
    }
    else {
        // For other surface types (seg xz, seg yz), clear them
        onSurfaceChanged(_surf_name, nullptr);
    }
}

void CVolumeViewer::onDrawingModeActive(bool active, float brushSize, bool isSquare)
{
    _drawingModeActive = active;
    _brushSize = brushSize;
    _brushIsSquare = isSquare;
    
    // Update the cursor to reflect the drawing mode state
    if (_cursor) {
        fScene->removeItem(_cursor);
        delete _cursor;
        _cursor = nullptr;
    }
    
    // Force cursor update
    POI *cursor = _surf_col->poi("cursor");
    if (cursor) {
        onPOIChanged("cursor", cursor);
    }
}

void CVolumeViewer::refreshPointPositions()
{
    if (!_point_collection) {
        return;
    }

    for (const auto& col_pair : _point_collection->getAllCollections()) {
        for (const auto& point_pair : col_pair.second.points) {
            if (_points_items.count(point_pair.first)) {
                renderOrUpdatePoint(point_pair.second);
            }
        }
    }
}
void CVolumeViewer::onPointAdded(const ColPoint& point)
{
    renderOrUpdatePoint(point);
}

void CVolumeViewer::onPointChanged(const ColPoint& point)
{
    renderOrUpdatePoint(point);
}

void CVolumeViewer::onPointRemoved(uint64_t pointId)
{
    if (_points_items.count(pointId)) {
        auto& pg = _points_items[pointId];
        fScene->removeItem(pg.circle);
        fScene->removeItem(pg.text);
        delete pg.circle;
        delete pg.text;
        _points_items.erase(pointId);
    }
}

void CVolumeViewer::onCollectionSelected(uint64_t collectionId)
{
    _selected_collection_id = collectionId;
}

void CVolumeViewer::onCollectionChanged(uint64_t collectionId)
{
    if (!_point_collection) {
        return;
    }

    const auto& collections = _point_collection->getAllCollections();
    auto it = collections.find(collectionId);
    if (it != collections.end()) {
        const auto& collection = it->second;
        for (const auto& point_pair : collection.points) {
            renderOrUpdatePoint(point_pair.second);
        }
    }
}

void CVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers modifiers)
{
    if (key == Qt::Key_Shift) {
        _new_shift_group_required = true;
    }
}

void CVolumeViewer::onPointSelected(uint64_t pointId)
{
    if (_selected_point_id == pointId) {
        return;
    }

    uint64_t old_selected_id = _selected_point_id;
    _selected_point_id = pointId;

    if (auto old_point = _point_collection->getPoint(old_selected_id)) {
        renderOrUpdatePoint(*old_point);
    }
    if (auto new_point = _point_collection->getPoint(_selected_point_id)) {
        renderOrUpdatePoint(*new_point);
    }
}

void CVolumeViewer::setResetViewOnSurfaceChange(bool reset)
{
    _resetViewOnSurfaceChange = reset;
}

void CVolumeViewer::updateAllOverlays()
{
    if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
        POI *poi = _surf_col->poi("focus");
        if (poi) {
            cv::Vec3f planeOrigin = plane->origin();
            // If plane origin differs from POI, update POI
            if (std::abs(poi->p[2] - planeOrigin[2]) > 0.01) {
                poi->p = planeOrigin;
                _surf_col->setPOI("focus", poi);  // NOW we do the expensive update
                emit sendZSliceChanged(static_cast<int>(poi->p[2]));
            }
        }
    }

    QPoint viewportPos = fGraphicsView->mapFromGlobal(QCursor::pos());
    QPointF scenePos = fGraphicsView->mapToScene(viewportPos);

    cv::Vec3f p, n;
    if (scene2vol(p, n, _surf, _surf_name, _surf_col, scenePos, _vis_center, _scale)) {
        POI *cursor = _surf_col->poi("cursor");
        if (!cursor)
            cursor = new POI;
        cursor->p = p;
        _surf_col->setPOI("cursor", cursor);
    }

    if (_point_collection && _dragged_point_id == 0) {
        uint64_t old_highlighted_id = _highlighted_point_id;
        _highlighted_point_id = 0;

        const float highlight_dist_threshold = 10.0f;
        float min_dist_sq = highlight_dist_threshold * highlight_dist_threshold;

        for (const auto& item_pair : _points_items) {
            auto item = item_pair.second.circle;
            QPointF point_scene_pos = item->rect().center();
            QPointF diff = scenePos - point_scene_pos;
            float dist_sq = QPointF::dotProduct(diff, diff);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                _highlighted_point_id = item_pair.first;
            }
        }

        if (old_highlighted_id != _highlighted_point_id) {
            if (auto old_point = _point_collection->getPoint(old_highlighted_id)) {
                renderOrUpdatePoint(*old_point);
            }
            if (auto new_point = _point_collection->getPoint(_highlighted_point_id)) {
                renderOrUpdatePoint(*new_point);
            }
        }
    }

    invalidateVis();
    invalidateIntersect();
    renderIntersections();
    renderDirectionHints();
    renderDirectionStepMarkers();
    renderPaths();
    refreshPointPositions();
}

void CVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    // Remove and delete existing items in the group
    clearOverlayGroup(key);
    _overlay_groups[key] = items;
}

// Visualize the 'step' parameter used by vc_grow_seg_from_segments by placing
// three small markers in either direction along the same direction arrows.
void CVolumeViewer::renderDirectionStepMarkers()
{
    if (!_showDirectionHints) {
        clearOverlayGroup("step_markers");
        return;
    }

    clearOverlayGroup("step_markers");

    auto* seg = dynamic_cast<QuadSurface*>(_surf_name == "segmentation" ? _surf : _surf_col->surface("segmentation"));
    if (!seg) return;

    // Determine step value and number of points
    QSettings settings("VC.ini", QSettings::IniFormat);
    bool use_seg_step = settings.value("viewer/use_seg_step_for_hints", true).toBool();
    int num_points = std::max(0, std::min(100, settings.value("viewer/direction_step_points", 5).toInt()));
    float step_val = settings.value("viewer/direction_step", 10.0).toFloat();
    if (use_seg_step && seg->meta) {
        try {
            if (seg->meta->contains("vc_grow_seg_from_segments_params")) {
                auto& p = seg->meta->at("vc_grow_seg_from_segments_params");
                if (p.contains("step")) step_val = p.at("step").get<float>();
            }
        } catch (...) {
            // keep settings default
        }
    }
    if (step_val <= 0) step_val = settings.value("viewer/direction_step", 10.0).toFloat();

    // Anchor at focus POI if possible
    cv::Vec3f target_wp;
    bool have_focus = false;
    if (auto* poi = _surf_col->poi("focus")) { target_wp = poi->p; have_focus = true; }

    std::vector<QGraphicsItem*> items;

    auto addDot = [&](const QPointF& center, const QColor& color, float radius = 3.0f) {
        auto* dot = new QGraphicsEllipseItem(center.x() - radius, center.y() - radius, 2*radius, 2*radius);
        dot->setPen(QPen(Qt::black, 1));
        dot->setBrush(QBrush(color));
        dot->setZValue(32);
        fScene->addItem(dot);
        items.push_back(dot);
    };

    if (_surf_name == "segmentation") {
        // Work in segmentation nominal coordinates converted to scene
        auto ptr = seg->pointer();
        if (have_focus) seg->pointTo(ptr, target_wp, 4.0, 100);
        cv::Vec3f nom = seg->loc(ptr) * _scale;
        // Center point
        addDot(QPointF(nom[0], nom[1]), QColor(255, 255, 0), 4.0f); // yellow center
        // Red side (+X)
        for (int n = 1; n <= num_points; ++n) {
            cv::Vec3f p = seg->loc(ptr, {n * step_val, 0, 0}) * _scale;
            addDot(QPointF(p[0], p[1]), Qt::red);
        }
        // Green side (−X)
        for (int n = 1; n <= num_points; ++n) {
            cv::Vec3f p = seg->loc(ptr, {-n * step_val, 0, 0}) * _scale;
            addDot(QPointF(p[0], p[1]), Qt::green);
        }
        setOverlayGroup("step_markers", items);
        return;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
        // Project segmentation step samples into plane view
        auto ptr = seg->pointer();
        if (have_focus) seg->pointTo(ptr, target_wp, 4.0, 100);
        cv::Vec3f p0 = seg->coord(ptr, {0,0,0});
        if (p0[0] == -1) return;
        cv::Vec3f s0 = plane->project(p0, 1.0f, _scale);
        addDot(QPointF(s0[0], s0[1]), QColor(255, 255, 0), 4.0f);

        for (int n = 1; n <= num_points; ++n) {
            cv::Vec3f p_pos = seg->coord(ptr, {n * step_val, 0, 0});
            cv::Vec3f p_neg = seg->coord(ptr, {-n * step_val, 0, 0});
            if (p_pos[0] != -1) {
                cv::Vec3f s = plane->project(p_pos, 1.0f, _scale);
                addDot(QPointF(s[0], s[1]), Qt::red);
            }
            if (p_neg[0] != -1) {
                cv::Vec3f s = plane->project(p_neg, 1.0f, _scale);
                addDot(QPointF(s[0], s[1]), Qt::green);
            }
        }
        setOverlayGroup("step_markers", items);
        return;
    }
}

void CVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlay_groups.find(key);
    if (it == _overlay_groups.end()) return;
    for (auto* item : it->second) {
        if (!item) continue;
        fScene->removeItem(item);
        delete item;
    }
    _overlay_groups.erase(it);
}

// Draw two small arrows indicating growth direction candidates:
// red = flip_x=false (along +X)
// green = flip_x=true (opposite −X)
// Shown on segmentation and projected into slice views.
void CVolumeViewer::renderDirectionHints()
{
    if (!_showDirectionHints) {
        clearOverlayGroup("direction_hints");
        return;
    }
    // Clear previous group
    clearOverlayGroup("direction_hints");

    if (!_surf) return;

    // Helper to create an arrow path item

    auto makeArrow = [&](const QPointF& origin, const QPointF& dir, const QColor& color) -> QGraphicsItem* {
        // Basic line with arrowhead
        const float line_len = 60.0f;      // scene units
        const float head_len = 10.0f;
        const float head_w   = 6.0f;

        // Normalize dir
        QPointF d = dir;
        double mag = std::hypot(d.x(), d.y());
        if (mag < 1e-3) mag = 1.0;
        d.setX(d.x()/mag); d.setY(d.y()/mag);

        QPointF tip = origin + QPointF(d.x()*line_len, d.y()*line_len);
        // Perpendicular for head
        QPointF perp(-d.y(), d.x());

        QPainterPath path;
        path.moveTo(origin);
        path.lineTo(tip);
        // Arrow head as a small V
        QPointF left  = tip - QPointF(d.x()*head_len, d.y()*head_len) + QPointF(perp.x()*head_w, perp.y()*head_w);
        QPointF right = tip - QPointF(d.x()*head_len, d.y()*head_len) - QPointF(perp.x()*head_w, perp.y()*head_w);
        path.moveTo(tip);
        path.lineTo(left);
        path.moveTo(tip);
        path.lineTo(right);

        auto* item = fGraphicsView->scene()->addPath(path, QPen(color, 2));
        item->setZValue(30); // Above intersections and points
        return item;
    };
    auto makeLabel = [&](const QPointF& pos, const QString& text, const QColor& color) -> QGraphicsItem* {
        auto* label = new COutlinedTextItem();
        label->setDefaultTextColor(color);
        label->setPlainText(text);
        // Make the label a bit smaller than default
        QFont f = label->font();
        f.setPointSizeF(9.0);
        label->setFont(f);
        label->setZValue(31);
        label->setPos(pos);
        fScene->addItem(label);
        return label;
    };

    if (_surf_name == "segmentation") {
        // Determine anchor in scene coords: prefer focus POI projected to segmentation; fallback to visible center
        QPointF anchor_scene = visible_center(fGraphicsView);

        if (auto* quad = dynamic_cast<QuadSurface*>(_surf)) {
            if (auto* poi = _surf_col->poi("focus")) {
                auto ptr = quad->pointer();
                float dist = quad->pointTo(ptr, poi->p, 4.0, 100);
                if (dist >= 0 && dist < 20.0/_scale) {
                    cv::Vec3f sp = quad->loc(ptr) * _scale;
                    anchor_scene = QPointF(sp[0], sp[1]);
                }
            }
        }

        // Offsets so the two arrows don't overlap the same origin point
        QPointF up_offset(0, -20.0);
        QPointF down_offset(0, 20.0);

        // On segmentation view, scene X is the surface +X direction.
        // User preference: green = flip_x=true (−X), red = flip_x=false (+X)
        QGraphicsItem* redArrow   = makeArrow(anchor_scene + up_offset, QPointF(1.0, 0.0), QColor(Qt::red));
        QGraphicsItem* greenArrow = makeArrow(anchor_scene + down_offset, QPointF(-1.0, 0.0), QColor(Qt::green));
        // Labels
        QGraphicsItem* redText   = makeLabel(anchor_scene + up_offset + QPointF(8, -8), QString("false"), QColor(Qt::red));
        QGraphicsItem* greenText = makeLabel(anchor_scene + down_offset + QPointF(8, -8), QString("true"), QColor(Qt::green));

        std::vector<QGraphicsItem*> items { redArrow, greenArrow, redText, greenText };
        setOverlayGroup("direction_hints", items);
        return;
    }

    // For slice plane views (seg xz / seg yz), project the segmentation +X tangent onto the plane and draw arrows
    if (auto* plane = dynamic_cast<PlaneSurface*>(_surf)) {
        auto* seg = dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation"));
        if (!seg) return;

        // Choose target world point near focus or plane origin
        cv::Vec3f target_wp = plane->origin();
        if (auto* poi = _surf_col->poi("focus")) {
            target_wp = poi->p;
        }

        // Find nearest point on segmentation and derive local +X tangent in 3D
        auto seg_ptr = seg->pointer();
        float dist = seg->pointTo(seg_ptr, target_wp, 4.0, 100);
        if (dist < 0) return;

        cv::Vec3f p0 = seg->coord(seg_ptr, {0,0,0});
        // Small nominal step along +X on the segmentation surface
        const float step_nominal = 2.0f;
        cv::Vec3f p1 = seg->coord(seg_ptr, {step_nominal, 0, 0});
        cv::Vec3f dir3 = p1 - p0;
        float len = std::sqrt(dir3.dot(dir3));
        if (len < 1e-5f) return;
        dir3 *= (1.0f / len);

        // Project to plane scene coordinates
        cv::Vec3f s0 = plane->project(p0, 1.0f, _scale);
        // Use a fixed scene length for the arrow
        const float scene_len = 60.0f;
        cv::Vec3f s1 = plane->project(p0 + dir3 * (scene_len / _scale), 1.0f, _scale);
        QPointF dir2(s1[0] - s0[0], s1[1] - s0[1]);
        double mag = std::hypot(dir2.x(), dir2.y());
        if (mag < 1e-3) return;

        QPointF anchor_scene(s0[0], s0[1]);
        // Slight offsets so they don't overlap exactly
        QPointF up_offset(0, -10.0);
        QPointF down_offset(0, 10.0);

        // User preference: green = flip_x=true (opposite of +X tangent), red = flip_x=false (along +X tangent)
        QGraphicsItem* redArrow   = makeArrow(anchor_scene + up_offset, dir2, QColor(Qt::red));
        QGraphicsItem* greenArrow = makeArrow(anchor_scene + down_offset, QPointF(-dir2.x(), -dir2.y()), QColor(Qt::green));
        // Labels near arrow tips
        QPointF redTip = anchor_scene + up_offset + QPointF(dir2.x(), dir2.y());
        QPointF greenTip = anchor_scene + down_offset + QPointF(-dir2.x(), -dir2.y());
        QGraphicsItem* redText   = makeLabel(redTip + QPointF(8, -8), QString("false"), QColor(Qt::red));
        QGraphicsItem* greenText = makeLabel(greenTip + QPointF(8, -8), QString("true"), QColor(Qt::green));

        std::vector<QGraphicsItem*> items { redArrow, greenArrow, redText, greenText };
        setOverlayGroup("direction_hints", items);
        return;
    }
}
