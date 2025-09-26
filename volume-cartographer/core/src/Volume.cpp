#include "vc/core/types/Volume.hpp"

#include <opencv2/imgcodecs.hpp>

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"


Volume::Volume(std::filesystem::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    _width = metadata_.get<int>("width");
    _height = metadata_.get<int>("height");
    _slices = metadata_.get<int>("slices");

    std::vector<std::mutex> init_mutexes(_slices);


    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name))
{
    metadata_.set("type", "vol");
    metadata_.set("width", _width);
    metadata_.set("height", _height);
    metadata_.set("slices", _slices);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});    

    zarrOpen();
}

void Volume::zarrOpen()
{
    if (!metadata_.hasKey("format") || metadata_.get<std::string>("format") != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);
    
    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::sort(groups.begin(), groups.end());
    
    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        z5::filesystem::handle::Dataset ds_handle(group, name, nlohmann::json::parse(std::ifstream(path_/name/".zarray")).value<std::string>("dimension_separator","."));

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);
    }
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path)
{
    return std::make_shared<Volume>(path);
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path, std::string uuid, std::string name)
{
    return std::make_shared<Volume>(path, uuid, name);
}

int Volume::sliceWidth() const { return _width; }
int Volume::sliceHeight() const { return _height; }
int Volume::numSlices() const { return _slices; }
double Volume::voxelSize() const
{
    return metadata_.get<double>("voxelsize");
}

z5::Dataset *Volume::zarrDataset(int level) const {
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}

namespace {

std::array<size_t, 3> spatialShapeTail(const std::vector<std::size_t>& shape)
{
    if (shape.size() < 3) {
        return {0, 0, 0};
    }
    // Treat the trailing dimensions as X, Y, Z respectively. OME-Zarr arrays
    // are typically laid out as (..., Z, Y, X) and we only care about the
    // spatial portion here.
    return {
        shape[shape.size() - 1], // X
        shape[shape.size() - 2], // Y
        shape[shape.size() - 3]  // Z
    };
}

double safeRatio(std::size_t numerator, std::size_t denominator)
{
    if (denominator == 0) {
        return 1.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

} // namespace

std::array<double, 3> Volume::levelScaleFactors(int level) const
{
    std::array<double, 3> factors{1.0, 1.0, 1.0};
    if (zarrDs_.empty() || level < 0 || level >= static_cast<int>(zarrDs_.size())) {
        return factors;
    }

    const auto baseShape = spatialShapeTail(zarrDs_.front()->shape());
    const auto levelShape = spatialShapeTail(zarrDs_[level]->shape());

    if (baseShape[0] == 0 || baseShape[1] == 0 || baseShape[2] == 0 ||
        levelShape[0] == 0 || levelShape[1] == 0 || levelShape[2] == 0) {
        return factors;
    }

    factors[0] = safeRatio(baseShape[0], levelShape[0]); // X axis downsample factor
    factors[1] = safeRatio(baseShape[1], levelShape[1]); // Y axis downsample factor
    factors[2] = safeRatio(baseShape[2], levelShape[2]); // Z axis downsample factor
    return factors;
}

std::array<double, 3> Volume::levelTranslations(int level) const
{
    (void)level;
    return {0.0, 0.0, 0.0};
}
