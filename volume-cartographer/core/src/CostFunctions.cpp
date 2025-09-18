#include "vc/core/util/CostFunctions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

using vc::core::util::NormalGridVolume;

namespace {
struct DistanceResult {
    double dist_sq = 0.0;
    std::array<double, 2> grad{0.0, 0.0};
};

DistanceResult point_line_dist_sq_grad(const std::array<double, 2>& p,
                                       const cv::Point2f& a,
                                       const cv::Point2f& b)
{
    DistanceResult res;

    double ab_x = static_cast<double>(b.x - a.x);
    double ab_y = static_cast<double>(b.y - a.y);
    double ap_x = p[0] - static_cast<double>(a.x);
    double ap_y = p[1] - static_cast<double>(a.y);
    double ab_len_sq = ab_x * ab_x + ab_y * ab_y;

    if (ab_len_sq < 1e-9) {
        double dx = ap_x;
        double dy = ap_y;
        res.dist_sq = dx * dx + dy * dy;
        res.grad[0] = 2.0 * dx;
        res.grad[1] = 2.0 * dy;
        return res;
    }

    double t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;
    double dt_dx = ab_x / ab_len_sq;
    double dt_dy = ab_y / ab_len_sq;

    if (t < 0.0) {
        t = 0.0;
        dt_dx = dt_dy = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
        dt_dx = dt_dy = 0.0;
    }

    double proj_x = static_cast<double>(a.x) + t * ab_x;
    double proj_y = static_cast<double>(a.y) + t * ab_y;
    double dx = p[0] - proj_x;
    double dy = p[1] - proj_y;

    res.dist_sq = dx * dx + dy * dy;
    res.grad[0] = 2.0 * dx * (1.0 - dt_dx * ab_x) + 2.0 * dy * (-dt_dx * ab_y);
    res.grad[1] = 2.0 * dx * (-dt_dy * ab_x) + 2.0 * dy * (1.0 - dt_dy * ab_y);

    return res;
}

constexpr std::array<std::array<int, 2>, 3> kPlaneAxes{{ {0, 1}, {0, 2}, {1, 2} }};
}

NormalConstraintPlane::AnalyticCostFunction::AnalyticCostFunction(const NormalGridVolume& normal_grid_volume,
                                                                  int plane_idx,
                                                                  double weight)
    : functor_(std::make_unique<NormalConstraintPlane>(normal_grid_volume, plane_idx, weight))
{
}

bool NormalConstraintPlane::AnalyticCostFunction::Evaluate(double const* const* parameters,
                                                           double* residuals,
                                                           double** jacobians) const
{
    std::array<std::array<double, 3>, 4> jacobian_blocks{};
    for (auto& block : jacobian_blocks) {
        block.fill(0.0);
    }

    double residual = 0.0;
    bool success = functor_->evaluate_analytic(parameters[0], parameters[1], parameters[2], parameters[3], &residual, &jacobian_blocks);

    residuals[0] = residual;

    if (jacobians) {
        for (int block = 0; block < 4; ++block) {
            if (jacobians[block]) {
                std::copy(jacobian_blocks[block].begin(), jacobian_blocks[block].end(), jacobians[block]);
            }
        }
    }

    (void)success; // The solver treats residual 0 with zero Jacobians as a no-op.
    return true;
}

bool NormalConstraintPlane::calculate_normal_snapping_loss_analytic(const std::array<double, 2>& p1,
                                                                    const std::array<double, 2>& p2,
                                                                    const cv::Point3f& query_point,
                                                                    int plane_idx,
                                                                    int grid_idx,
                                                                    double* out_loss,
                                                                    std::array<double, 2>* grad_p1,
                                                                    std::array<double, 2>* grad_p2) const
{
    grad_p1->fill(0.0);
    grad_p2->fill(0.0);
    *out_loss = 0.0;

    double edge_vec_x = p2[0] - p1[0];
    double edge_vec_y = p2[1] - p1[1];
    double edge_len_sq = edge_vec_x * edge_vec_x + edge_vec_y * edge_vec_y;
    double edge_len = std::sqrt(edge_len_sq);
    if (edge_len < 1e-9) {
        return false;
    }

    double inv_len = 1.0 / edge_len;
    double inv_len_cubed = inv_len / edge_len_sq;

    double edge_normal_x = edge_vec_y * inv_len;
    double edge_normal_y = -edge_vec_x * inv_len;

    double dnx_ddx = -edge_vec_y * edge_vec_x * inv_len_cubed;
    double dnx_ddy = inv_len - edge_vec_y * edge_vec_y * inv_len_cubed;
    double dny_ddx = -inv_len + edge_vec_x * edge_vec_x * inv_len_cubed;
    double dny_ddy = edge_vec_x * edge_vec_y * inv_len_cubed;

    cv::Point2f p1_cv(static_cast<float>(p1[0]), static_cast<float>(p1[1]));
    cv::Point2f p2_cv(static_cast<float>(p2[0]), static_cast<float>(p2[1]));

    auto& path_cache = path_caches_[grid_idx];
    if (!path_cache.valid(p1_cv, p2_cv, cache_radius_normal_, &normal_grid_volume)) {
        PathCachePayload payload;
        if (const auto* grid = normal_grid_volume.query_nearest(query_point, plane_idx)) {
            cv::Point2f midpoint_cv = 0.5f * (p1_cv + p2_cv);
            payload = grid->get(midpoint_cv, query_radius_);
        }
        path_cache.put(p1_cv, p2_cv, std::move(payload), &normal_grid_volume);
        snap_loss_caches_[grid_idx] = {};
    }
    const auto& nearby_paths = path_cache.get();
    if (nearby_paths.empty()) {
        return false;
    }

    double total_weighted_dot_product = 0.0;
    double total_weight = 0.0;
    std::array<double, 2> grad_normal_p1{0.0, 0.0};
    std::array<double, 2> grad_normal_p2{0.0, 0.0};

    for (const auto& path_ptr : nearby_paths) {
        const auto& path = *path_ptr;
        if (path.size() < 2) continue;

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            cv::Point2f p_a = path[i];
            cv::Point2f p_b = path[i + 1];

            float dist_sq = seg_dist_sq_appx(p1_cv, p2_cv, p_a, p_b);
            if (dist_sq > roi_radius_ * roi_radius_) {
                continue;
            }
            dist_sq = std::max(0.1f, dist_sq);

            double weight_n = 1.0 / dist_sq;

            cv::Point2f tangent = p_b - p_a;
            float length = cv::norm(tangent);
            if (length > 0.0f) {
                tangent /= length;
            }
            cv::Point2f normal(-tangent.y, tangent.x);

            double dot_product = edge_normal_x * normal.x + edge_normal_y * normal.y;
            double sign_dot = dot_product >= 0.0 ? 1.0 : -1.0;
            double abs_dot = std::abs(dot_product);

            total_weighted_dot_product += weight_n * abs_dot;
            total_weight += weight_n;

            double common_x = weight_n * sign_dot * (dnx_ddx * normal.x + dny_ddx * normal.y);
            double common_y = weight_n * sign_dot * (dnx_ddy * normal.x + dny_ddy * normal.y);

            grad_normal_p1[0] -= common_x;
            grad_normal_p1[1] -= common_y;
            grad_normal_p2[0] += common_x;
            grad_normal_p2[1] += common_y;
        }
    }

    double normal_loss = 0.0;
    if (total_weight > 1e-9) {
        normal_loss = 1.0 - total_weighted_dot_product / total_weight;
        double inv_total_weight = 1.0 / total_weight;
        grad_normal_p1[0] *= -inv_total_weight;
        grad_normal_p1[1] *= -inv_total_weight;
        grad_normal_p2[0] *= -inv_total_weight;
        grad_normal_p2[1] *= -inv_total_weight;
    } else {
        grad_normal_p1.fill(0.0);
        grad_normal_p2.fill(0.0);
    }

    auto& snap_cache = snap_loss_caches_[grid_idx];
    if (!snap_cache.valid(p1_cv, p2_cv, cache_radius_snap_, &normal_grid_volume)) {
        SnapLossPayload payload;
        float closest_dist_norm = std::numeric_limits<float>::max();

        for (int path_idx = 0; path_idx < static_cast<int>(nearby_paths.size()); ++path_idx) {
            const auto& path = *nearby_paths[path_idx];
            if (path.size() < 2) continue;

            for (int i = 0; i < static_cast<int>(path.size()) - 1; ++i) {
                float d2_sq = point_line_dist_sq(p2_cv, path[i], path[i + 1]);
                if (d2_sq >= snap_trig_th_ * snap_trig_th_) continue;

                if (i < static_cast<int>(path.size()) - 2) {
                    float d1_sq = point_line_dist_sq(p1_cv, path[i + 1], path[i + 2]);
                    if (d1_sq < snap_search_range_ * snap_search_range_) {
                        float dist_norm = 0.5f * (std::sqrt(d1_sq) / snap_search_range_ + std::sqrt(d2_sq) / snap_trig_th_);
                        if (dist_norm < closest_dist_norm) {
                            closest_dist_norm = dist_norm;
                            payload.best_path_idx = path_idx;
                            payload.best_seg_idx = i;
                            payload.best_is_next = true;
                        }
                    }
                }
                if (i > 0) {
                    float d1_sq = point_line_dist_sq(p1_cv, path[i - 1], path[i]);
                    if (d1_sq < snap_search_range_ * snap_search_range_) {
                        float dist_norm = 0.5f * (std::sqrt(d1_sq) / snap_search_range_ + std::sqrt(d2_sq) / snap_trig_th_);
                        if (dist_norm < closest_dist_norm) {
                            closest_dist_norm = dist_norm;
                            payload.best_path_idx = path_idx;
                            payload.best_seg_idx = i;
                            payload.best_is_next = false;
                        }
                    }
                }
            }
        }
        snap_cache.put(p1_cv, p2_cv, payload, &normal_grid_volume);
    }

    const auto& snap_payload = snap_cache.get();
    double snapping_loss = 1.0;
    std::array<double, 2> grad_snap_p1{0.0, 0.0};
    std::array<double, 2> grad_snap_p2{0.0, 0.0};

    if (snap_payload.best_path_idx != -1) {
        const auto& best_path = *nearby_paths[snap_payload.best_path_idx];
        const auto& seg2_p1 = best_path[snap_payload.best_seg_idx];
        const auto& seg2_p2 = best_path[snap_payload.best_seg_idx + 1];

        cv::Point2f seg1_p1;
        cv::Point2f seg1_p2;
        if (snap_payload.best_is_next) {
            seg1_p1 = best_path[snap_payload.best_seg_idx + 1];
            seg1_p2 = best_path[snap_payload.best_seg_idx + 2];
        } else {
            seg1_p1 = best_path[snap_payload.best_seg_idx - 1];
            seg1_p2 = best_path[snap_payload.best_seg_idx];
        }

        DistanceResult d1 = point_line_dist_sq_grad(p1, seg1_p1, seg1_p2);
        DistanceResult d2 = point_line_dist_sq_grad(p2, seg2_p1, seg2_p2);

        double d1_norm = 0.0;
        std::array<double, 2> grad_d1_norm{0.0, 0.0};
        if (d1.dist_sq < 1e-9) {
            d1_norm = d1.dist_sq / snap_search_range_;
            grad_d1_norm[0] = d1.grad[0] / snap_search_range_;
            grad_d1_norm[1] = d1.grad[1] / snap_search_range_;
        } else {
            double scale = 0.5 / (std::sqrt(d1.dist_sq) * snap_search_range_);
            d1_norm = std::sqrt(d1.dist_sq) / snap_search_range_;
            grad_d1_norm[0] = d1.grad[0] * scale;
            grad_d1_norm[1] = d1.grad[1] * scale;
        }

        double d2_norm = 0.0;
        std::array<double, 2> grad_d2_norm{0.0, 0.0};
        if (d2.dist_sq < 1e-9) {
            d2_norm = d2.dist_sq / snap_trig_th_;
            grad_d2_norm[0] = d2.grad[0] / snap_trig_th_;
            grad_d2_norm[1] = d2.grad[1] / snap_trig_th_;
        } else {
            double scale = 0.5 / (std::sqrt(d2.dist_sq) * snap_trig_th_);
            d2_norm = std::sqrt(d2.dist_sq) / snap_trig_th_;
            grad_d2_norm[0] = d2.grad[0] * scale;
            grad_d2_norm[1] = d2.grad[1] * scale;
        }

        snapping_loss = d1_norm * (1.0 - d2_norm) + d2_norm;
        grad_snap_p1[0] = grad_d1_norm[0] * (1.0 - d2_norm);
        grad_snap_p1[1] = grad_d1_norm[1] * (1.0 - d2_norm);
        grad_snap_p2[0] = grad_d2_norm[0] * (1.0 - d1_norm);
        grad_snap_p2[1] = grad_d2_norm[1] * (1.0 - d1_norm);
    }

    *grad_p1 = grad_normal_p1;
    (*grad_p1)[0] += 0.1 * grad_snap_p1[0];
    (*grad_p1)[1] += 0.1 * grad_snap_p1[1];

    *grad_p2 = grad_normal_p2;
    (*grad_p2)[0] += 0.1 * grad_snap_p2[0];
    (*grad_p2)[1] += 0.1 * grad_snap_p2[1];

    *out_loss = normal_loss + 0.1 * snapping_loss;
    return true;
}

bool NormalConstraintPlane::evaluate_analytic(const double* pA,
                                              const double* pB1,
                                              const double* pB2,
                                              const double* pC,
                                              double* residual,
                                              std::array<std::array<double, 3>, 4>* jacobians) const
{
    residual[0] = 0.0;
    for (auto& block : *jacobians) {
        block.fill(0.0);
    }

    const int normal_axis = 2 - plane_idx;

    double a_coord = pA[normal_axis];
    double b1_rel = pB1[normal_axis] - a_coord;
    double b2_rel = pB2[normal_axis] - a_coord;
    double c_rel = pC[normal_axis] - a_coord;

    const double* pBn = nullptr;
    int bn_index = -1;

    if (std::abs(c_rel) < 1e-9) {
        if (std::abs(b1_rel) > 1e-9) {
            pBn = pB1;
            bn_index = 1;
        } else if (std::abs(b2_rel) > 1e-9) {
            pBn = pB2;
            bn_index = 2;
        }
    } else {
        if (c_rel > 0) {
            if (b1_rel <= 0) {
                pBn = pB1;
                bn_index = 1;
            } else if (b2_rel <= 0) {
                pBn = pB2;
                bn_index = 2;
            }
        } else {
            if (b1_rel >= 0) {
                pBn = pB1;
                bn_index = 1;
            } else if (b2_rel >= 0) {
                pBn = pB2;
                bn_index = 2;
            }
        }
    }

    if (!pBn) {
        return false;
    }

    double denominator = pBn[normal_axis] - pC[normal_axis];
    if (std::abs(denominator) < 1e-9) {
        return false;
    }

    double t = -c_rel / denominator;

    std::array<double, 3> pE{};
    std::array<double, 3> bn_minus_c{};
    for (int i = 0; i < 3; ++i) {
        bn_minus_c[i] = pBn[i] - pC[i];
        pE[i] = pC[i] + t * bn_minus_c[i];
    }

    const auto& axes = kPlaneAxes[plane_idx];
    std::array<double, 2> pA2{pA[axes[0]], pA[axes[1]]};
    std::array<double, 2> pE2{pE[axes[0]], pE[axes[1]]};

    cv::Point3f query_point(static_cast<float>(pA[0]), static_cast<float>(pA[1]), static_cast<float>(pA[2]));

    double interpolated_loss = 0.0;
    std::array<double, 2> grad_p1{0.0, 0.0};
    std::array<double, 2> grad_p2{0.0, 0.0};
    if (!calculate_normal_snapping_loss_analytic(pA2, pE2, query_point, plane_idx, 0, &interpolated_loss, &grad_p1, &grad_p2)) {
        return false;
    }

    std::array<double, 3> grad_L_pA{0.0, 0.0, 0.0};
    std::array<double, 3> grad_L_pBn{0.0, 0.0, 0.0};
    std::array<double, 3> grad_L_pC{0.0, 0.0, 0.0};
    std::array<double, 3> grad_L_pB1{0.0, 0.0, 0.0};
    std::array<double, 3> grad_L_pB2{0.0, 0.0, 0.0};

    grad_L_pA[axes[0]] += grad_p1[0];
    grad_L_pA[axes[1]] += grad_p1[1];

    double den = denominator;
    double dtdpA = 1.0 / den;
    double dtdpBn = c_rel / (den * den);
    double dtdpC = -1.0 / den + c_rel / (den * den);

    std::array<std::array<double, 3>, 3> dpE_dA{};
    std::array<std::array<double, 3>, 3> dpE_dBn{};
    std::array<std::array<double, 3>, 3> dpE_dC{};

    for (int i = 0; i < 3; ++i) {
        dpE_dA[i][normal_axis] = dtdpA * bn_minus_c[i];

        dpE_dBn[i][i] += t;
        dpE_dBn[i][normal_axis] += dtdpBn * bn_minus_c[i];

        dpE_dC[i][i] += 1.0 - t;
        dpE_dC[i][normal_axis] += dtdpC * bn_minus_c[i];
    }

    for (int k = 0; k < 2; ++k) {
        int axis = axes[k];
        double g = grad_p2[k];
        for (int j = 0; j < 3; ++j) {
            grad_L_pA[j] += g * dpE_dA[axis][j];
            grad_L_pBn[j] += g * dpE_dBn[axis][j];
            grad_L_pC[j] += g * dpE_dC[axis][j];
        }
    }

    if (bn_index == 1) {
        grad_L_pB1 = grad_L_pBn;
    } else {
        grad_L_pB2 = grad_L_pBn;
    }

    std::array<double, 3> v_abn{pBn[0] - pA[0], pBn[1] - pA[1], pBn[2] - pA[2]};
    std::array<double, 3> v_ac{pC[0] - pA[0], pC[1] - pA[1], pC[2] - pA[2]};

    std::array<double, 3> cross{
        v_abn[1] * v_ac[2] - v_abn[2] * v_ac[1],
        v_abn[2] * v_ac[0] - v_abn[0] * v_ac[2],
        v_abn[0] * v_ac[1] - v_abn[1] * v_ac[0]
    };

    double cross_len_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    double cross_len = std::sqrt(cross_len_sq);
    double den_angle = cross_len + 1e-9;
    double cos_angle = cross[normal_axis] / den_angle;
    double angle_weight = 1.0 - std::abs(cos_angle);
    double sign_cos = cos_angle >= 0.0 ? 1.0 : -1.0;

    std::array<std::array<double, 3>, 3> dCross_dVabn{{
        {0.0, v_ac[2], -v_ac[1]},
        {-v_ac[2], 0.0, v_ac[0]},
        {v_ac[1], -v_ac[0], 0.0}
    }};

    std::array<std::array<double, 3>, 3> dCross_dVac{{
        {0.0, -v_abn[2], v_abn[1]},
        {v_abn[2], 0.0, -v_abn[0]},
        {-v_abn[1], v_abn[0], 0.0}
    }};

    auto compute_crosslen_grad = [&](const std::array<double, 3>& dCross) {
        if (cross_len < 1e-12) {
            return 0.0;
        }
        return (cross[0] * dCross[0] + cross[1] * dCross[1] + cross[2] * dCross[2]) / cross_len;
    };

    std::array<double, 3> grad_angle_pA{0.0, 0.0, 0.0};
    std::array<double, 3> grad_angle_pB1{0.0, 0.0, 0.0};
    std::array<double, 3> grad_angle_pB2{0.0, 0.0, 0.0};
    std::array<double, 3> grad_angle_pC{0.0, 0.0, 0.0};

    for (int axis = 0; axis < 3; ++axis) {
        std::array<double, 3> dCross_dBn = dCross_dVabn[axis];
        std::array<double, 3> dCross_dC = dCross_dVac[axis];
        std::array<double, 3> dCross_dA{
            -dCross_dBn[0] - dCross_dC[0],
            -dCross_dBn[1] - dCross_dC[1],
            -dCross_dBn[2] - dCross_dC[2]
        };

        auto accumulate = [&](std::array<double, 3>& target, const std::array<double, 3>& dCross) {
            double dCross_n = dCross[normal_axis];
            double dCross_len = compute_crosslen_grad(dCross);
            double dCos = (dCross_n * den_angle - cross[normal_axis] * dCross_len) / (den_angle * den_angle);
            double dAngle = -sign_cos * dCos;
            target[axis] = dAngle;
        };

        accumulate(grad_angle_pA, dCross_dA);
        if (bn_index == 1) {
            accumulate(grad_angle_pB1, dCross_dBn);
        } else {
            accumulate(grad_angle_pB2, dCross_dBn);
        }
        accumulate(grad_angle_pC, dCross_dC);
    }

    double weight_term = weight;
    double total_loss = interpolated_loss;
    double residual_value = weight_term * total_loss * angle_weight;

    (*jacobians)[0][0] = weight_term * (angle_weight * grad_L_pA[0] + total_loss * grad_angle_pA[0]);
    (*jacobians)[0][1] = weight_term * (angle_weight * grad_L_pA[1] + total_loss * grad_angle_pA[1]);
    (*jacobians)[0][2] = weight_term * (angle_weight * grad_L_pA[2] + total_loss * grad_angle_pA[2]);

    (*jacobians)[1][0] = weight_term * (angle_weight * grad_L_pB1[0] + total_loss * grad_angle_pB1[0]);
    (*jacobians)[1][1] = weight_term * (angle_weight * grad_L_pB1[1] + total_loss * grad_angle_pB1[1]);
    (*jacobians)[1][2] = weight_term * (angle_weight * grad_L_pB1[2] + total_loss * grad_angle_pB1[2]);

    (*jacobians)[2][0] = weight_term * (angle_weight * grad_L_pB2[0] + total_loss * grad_angle_pB2[0]);
    (*jacobians)[2][1] = weight_term * (angle_weight * grad_L_pB2[1] + total_loss * grad_angle_pB2[1]);
    (*jacobians)[2][2] = weight_term * (angle_weight * grad_L_pB2[2] + total_loss * grad_angle_pB2[2]);

    (*jacobians)[3][0] = weight_term * (angle_weight * grad_L_pC[0] + total_loss * grad_angle_pC[0]);
    (*jacobians)[3][1] = weight_term * (angle_weight * grad_L_pC[1] + total_loss * grad_angle_pC[1]);
    (*jacobians)[3][2] = weight_term * (angle_weight * grad_L_pC[2] + total_loss * grad_angle_pC[2]);

    residual[0] = residual_value;
    return true;
}
