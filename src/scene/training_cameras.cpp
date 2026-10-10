#include "scene/training_cameras.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace gsr {

float TrainingCamera::fovYDeg() const {
    return float(2.0 * std::atan(double(height) / (2.0 * double(fy))) * 180.0 / 3.14159265358979323846);
}

std::vector<TrainingCamera> loadTrainingCameras(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open cameras file: " + std::string(reinterpret_cast<const char*>(path.u8string().c_str())));
    std::vector<TrainingCamera> cameras;
    try {
        const nlohmann::json j = nlohmann::json::parse(file);
        if (!j.is_array()) throw std::runtime_error("expected a JSON array");
        for (const nlohmann::json& c : j) {
            TrainingCamera cam;
            cam.id = c.at("id").get<int>();
            cam.imageName = c.value("img_name", "");
            cam.width = c.at("width").get<uint32_t>();
            cam.height = c.at("height").get<uint32_t>();
            cam.fx = c.at("fx").get<float>();
            cam.fy = c.at("fy").get<float>();
            const nlohmann::json& p = c.at("position");
            const nlohmann::json& r = c.at("rotation");
            if (p.size() != 3 || r.size() != 3) throw std::runtime_error("bad position/rotation size");
            cam.position = {p[0].get<float>(), p[1].get<float>(), p[2].get<float>()};
            // JSON rows -> GLM columns: element (row, col) goes to cameraToWorld[col][row].
            for (int row = 0; row < 3; ++row) {
                if (r[row].size() != 3) throw std::runtime_error("bad rotation row");
                for (int col = 0; col < 3; ++col) cam.cameraToWorld[col][row] = r[row][col].get<float>();
            }
            if (cam.width == 0 || cam.height == 0 || !(cam.fy > 0.0f))
                throw std::runtime_error("bad intrinsics for camera " + std::to_string(cam.id));
            cameras.push_back(std::move(cam));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid cameras file " + std::string(reinterpret_cast<const char*>(path.u8string().c_str())) + ": " + e.what());
    }
    return cameras;
}

glm::vec3 meanCameraUp(const std::vector<TrainingCamera>& cameras) {
    glm::dvec3 sum(0.0);
    for (const TrainingCamera& c : cameras) sum -= glm::dvec3(c.down());
    const double len = glm::length(sum);
    return len > 1e-9 ? glm::vec3(sum / len) : glm::vec3(0.0f, -1.0f, 0.0f);
}

glm::vec3 cameraFocusPoint(const std::vector<TrainingCamera>& cameras) {
    // Minimize sum_i |(I - f_i f_i^T)(p - c_i)|^2 (squared distance from p to each ray's
    // line). Normal equations: [sum (I - f f^T)] p = sum (I - f f^T) c.
    glm::dmat3 A(0.0);
    glm::dvec3 b(0.0), centroid(0.0);
    for (const TrainingCamera& c : cameras) {
        const glm::dvec3 f = glm::normalize(glm::dvec3(c.forward()));
        const glm::dmat3 P = glm::dmat3(1.0) - glm::outerProduct(f, f);
        A += P;
        b += P * glm::dvec3(c.position);
        centroid += glm::dvec3(c.position);
    }
    if (cameras.empty()) return glm::vec3(0.0f);
    centroid /= double(cameras.size());
    // Parallel axes (e.g. a forward-facing capture) make A singular: no unique point.
    if (std::fabs(glm::determinant(A)) < 1e-6 * double(cameras.size())) return glm::vec3(centroid);
    return glm::vec3(glm::inverse(A) * b);
}

}  // namespace gsr
