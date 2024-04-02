//
// Dingdong Yang, 04/01/2023
//

#include <cfloat>
#include <filesystem>
#include <vector>
#include <fstream>
#include <string>

#include <torch/extension.h>

namespace fs = std::filesystem;

std::vector<std::string> simpleGlob(const fs::path& search_path, const std::string& regex_pattern) {
    std::vector<std::string> matched_files;
    std::regex pattern(regex_pattern, std::regex_constants::basic);

    if (!fs::exists(search_path) || !fs::is_directory(search_path)) {
        std::cerr << "Path is not a directory or does not exist: " << search_path << std::endl;
        return matched_files;
    }

    for (const auto& entry : fs::directory_iterator(search_path)) {
        if (fs::is_regular_file(entry) && std::regex_search(entry.path().filename().string(), pattern)) {
            matched_files.push_back(entry.path().string());
        }
    }

    return matched_files;
}


float getMaximumVoxelLength(std::string dataRoot, const int level0UnitLength = 361){
    std::vector<std::string> allDataPaths = simpleGlob(dataRoot, ".*\\.bin");
    float maxVL = 0.f;
    for (auto &path: allDataPaths){
        std::ifstream file(path, std::ios::in | std::ios::binary);
        float octreeRootNumFloat;
        file.read(reinterpret_cast<char*>(&octreeRootNumFloat), sizeof(float));
        int octreeRootNum = static_cast<int>(octreeRootNumFloat);
        float lengthf;
        file.read(reinterpret_cast<char*>(&lengthf), sizeof(float));
        file.seekg(sizeof(float) * 2, std::ios::cur);
        int length = static_cast<int>(lengthf);

        for (int i = 0; i < length; i++){
            auto data = new float[level0UnitLength];
            file.read(reinterpret_cast<char*>(data), sizeof(float) * level0UnitLength);
            maxVL = std::max(maxVL, data[level0UnitLength - 7]);
            delete[] data;
        }
    }
    return maxVL;
}

void loadFromFile(std::ifstream& file,
                  torch::Tensor &out,
                  const int length,
                  const int unitLength,
                  const bool rootFlag){

    for (int i = 0; i < length; i++){
        auto data = new float[unitLength];
        file.read(reinterpret_cast<char*>(data), sizeof(float) * unitLength);
        int parentIdx = rootFlag ? 0 : static_cast<int>(data[1]);
        int currentIdx = static_cast<int>(data[0]);
        torch::Tensor nodeVec = torch::from_blob(data + 2, {unitLength - 2}).clone();
        out.index_put_({parentIdx * 8 + currentIdx, torch::indexing::None}, nodeVec);
        delete[] data;
    }
}

std::vector<torch::Tensor> readAndConstruct(std::string inputPath,
                                            const int level0UnitLength = 361,
                                            const int level1UnitLength = 139,
                                            const int level2UnitLength = 139){
    std::ifstream file(inputPath, std::ios::in | std::ios::binary);

    float octreeRootNumFloat;
    file.read(reinterpret_cast<char*>(&octreeRootNumFloat), sizeof(float));
    int octreeRootNum = static_cast<int>(octreeRootNumFloat);
    float length0f, length1f, length2f;
    file.read(reinterpret_cast<char*>(&length0f), sizeof(float));
    file.read(reinterpret_cast<char*>(&length1f), sizeof(float));
    file.read(reinterpret_cast<char*>(&length2f), sizeof(float));
    int length0, length1, length2;
    length0 = static_cast<int>(length0f);
    length1 = static_cast<int>(length1f);
    length2 = static_cast<int>(length2f);

    // The deducted 2 are the tree topology indicators (parent ID and children ID)
    torch::Tensor level0Tensor = torch::zeros({octreeRootNum, level0UnitLength - 2});
    torch::Tensor level1Tensor = torch::zeros({octreeRootNum * 8, level1UnitLength - 2});
    torch::Tensor level2Tensor = torch::zeros({octreeRootNum * 8 * 8, level2UnitLength - 2});
    loadFromFile(file, level0Tensor, length0, level0UnitLength, true);
    loadFromFile(file, level1Tensor, length1, level1UnitLength, false);
    loadFromFile(file, level2Tensor, length2, level2UnitLength, false);

    return {level0Tensor, level1Tensor, level2Tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load", &readAndConstruct, "Read binary and construct tree tensors");
  m.def("max_voxel_length", &getMaximumVoxelLength,
        "Get the maximum root voxel length for normalization.");
}