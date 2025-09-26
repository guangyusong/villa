#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <optional>
#include <limits>

#include "vc/core/util/Surface.hpp"

struct OrientedBBox
{
    cv::Vec3f center{0.f, 0.f, 0.f};
    cv::Vec3f halfExtents{0.f, 0.f, 0.f}; // along axisU, axisV, axisN
    cv::Vec3f axisU{1.f, 0.f, 0.f};
    cv::Vec3f axisV{0.f, 1.f, 0.f};
    cv::Vec3f axisN{0.f, 0.f, 1.f};

    bool valid() const
    {
        return halfExtents[0] > 0.f && halfExtents[1] > 0.f && halfExtents[2] >= 0.f;
    }
};

inline OrientedBBox makeAxisAlignedBBox(const Rect3D& rect)
{
    OrientedBBox box;
    box.center = 0.5f * (rect.low + rect.high);
    box.halfExtents = 0.5f * (rect.high - rect.low);
    box.axisU = {1.f, 0.f, 0.f};
    box.axisV = {0.f, 1.f, 0.f};
    box.axisN = {0.f, 0.f, 1.f};
    return box;
}

inline Rect3D orientedBBoxToRect(const OrientedBBox& box)
{
    std::array<cv::Vec3f, 8> corners;
    int idx = 0;
    for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
            for (int sz : {-1, 1}) {
                corners[idx++] = box.center
                    + box.axisU * (box.halfExtents[0] * static_cast<float>(sx))
                    + box.axisV * (box.halfExtents[1] * static_cast<float>(sy))
                    + box.axisN * (box.halfExtents[2] * static_cast<float>(sz));
            }
        }
    }

    Rect3D rect;
    rect.low = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    rect.high = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    for (const auto& corner : corners) {
        for (int axis = 0; axis < 3; ++axis) {
            rect.low[axis] = std::min(rect.low[axis], corner[axis]);
            rect.high[axis] = std::max(rect.high[axis], corner[axis]);
        }
    }
    return rect;
}

inline cv::Vec3f axisIndexToUnit(int axisIndex)
{
    switch (axisIndex) {
    case 0: return {1.f, 0.f, 0.f};
    case 1: return {0.f, 1.f, 0.f};
    case 2: return {0.f, 0.f, 1.f};
    default: return {0.f, 0.f, 0.f};
    }
}
