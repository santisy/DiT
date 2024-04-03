//
// Dingdong Yang, 04/01/2023
//

#include <cfloat>
#include <cstring>
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
        file.close();
    }
    return maxVL;
}

void loadFromFile(std::ifstream& file,
                  at::Tensor &out,
                  const int length,
                  const int unitLength,
                  const bool rootFlag){

    for (int i = 0; i < length; i++){
        auto data = new float[unitLength];
        file.read(reinterpret_cast<char*>(data), sizeof(float) * unitLength);
        int parentIdx = rootFlag ? 0 : static_cast<int>(data[1]);
        int currentIdx = static_cast<int>(data[0]);
        at::Tensor nodeVec = at::from_blob(data + 2, {unitLength - 2}).clone();
        out.index_put_({parentIdx * 8 + currentIdx, at::indexing::Slice()}, nodeVec);
        delete[] data;
    }
}

void loadFromFileAndAssignPos(std::ifstream& file,
                              at::Tensor &preScales,
                              at::Tensor &outScales,
                              at::Tensor &out,
                              at::Tensor &prevPos,
                              at::Tensor &outPos,
                              const int length,
                              const int unitLength){

    for (int i = 0; i < length; i++){
        auto data = new float[unitLength];
        file.read(reinterpret_cast<char*>(data), sizeof(float) * unitLength);
        int parentIdx = static_cast<int>(data[1]);
        int currentIdx = static_cast<int>(data[0]);
        int outIdx = parentIdx * 8 + currentIdx;

        at::Tensor absoluteScale = preScales.index({parentIdx}) * 0.5f * 1.05f;
        outScales.index_put_({outIdx}, absoluteScale);

        int zIdx = (currentIdx >> 2);
        int yIdx = ((currentIdx - zIdx * 4) >> 1);
        int xIdx = currentIdx - zIdx * 4 - yIdx * 2;
        at::Tensor scale = preScales.index({parentIdx}) * 0.5f;

        // Deduce the children positions as the children voxel center
        outPos.index_put_({outIdx, 0}, prevPos.index({parentIdx, 0}) - scale + scale * 0.95f * xIdx + absoluteScale / 2.0f);
        outPos.index_put_({outIdx, 1}, prevPos.index({parentIdx, 1}) - scale + scale * 0.95f * yIdx + absoluteScale / 2.0f);
        outPos.index_put_({outIdx, 2}, prevPos.index({parentIdx, 2}) - scale + scale * 0.95f * zIdx + absoluteScale / 2.0f);

        at::Tensor nodeVec = at::from_blob(data + 2, {unitLength - 2}).clone();
        out.index_put_({outIdx, at::indexing::Slice()}, nodeVec);

        delete[] data;
    }
}

std::vector<at::Tensor> readAndConstruct(std::string inputPath,
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
    at::Tensor level0Tensor = at::zeros({octreeRootNum, level0UnitLength - 2});
    at::Tensor level1Tensor = at::zeros({octreeRootNum * 8, level1UnitLength - 2});
    at::Tensor level2Tensor = at::zeros({octreeRootNum * 8 * 8, level2UnitLength - 2});

    // Positions
    at::Tensor level1Positions = at::zeros({octreeRootNum * 8, 3});
    at::Tensor level2Positions = at::zeros({octreeRootNum * 8 * 8, 3});
    // Scales
    at::Tensor level1Scales = at::zeros({octreeRootNum * 8});
    at::Tensor level2Scales = at::zeros({octreeRootNum * 8 * 8});

    loadFromFile(file, level0Tensor, length0, level0UnitLength, true);
    at::Tensor level0Positions = level0Tensor.index({at::indexing::Slice(), at::indexing::Slice(-3, at::indexing::None)}).clone();
    at::Tensor level0Scales = level0Tensor.index({at::indexing::Slice(), -7}).clone();
    loadFromFileAndAssignPos(file, level0Scales, level1Scales, level1Tensor, level0Positions, level1Positions, length1, level1UnitLength);
    loadFromFileAndAssignPos(file, level1Scales, level2Scales, level2Tensor, level1Positions, level2Positions, length2, level2UnitLength);

    file.close();

    return {level0Tensor, level1Tensor, level2Tensor,
            level0Positions, level1Positions, level2Positions};
}

at::Tensor deducePositionFromSample(at::Tensor preScales,
                                    at::Tensor &outScales,
                                    at::Tensor &prevPos,
                                    const int length){
    at::Tensor outPos = at::zeros({preScales.size(0), length, 3}).to(preScales.device());
    for (int i = 0; i < length; i++){
        int parentIdx = i / 8;
        int currentIdx = i % 8;
        int outIdx = parentIdx * 8 + currentIdx;

        at::Tensor absoluteScale = preScales.index({at::indexing::Slice(), parentIdx}) * 0.5f * 1.05f;
        outScales.index_put_({at::indexing::Slice(), outIdx}, absoluteScale);

        int zIdx = (currentIdx >> 2);
        int yIdx = ((currentIdx - zIdx * 4) >> 1);
        int xIdx = currentIdx - zIdx * 4 - yIdx * 2;
        at::Tensor scale = preScales.index({at::indexing::Slice(), parentIdx}) * 0.5f;

        // Deduce the children positions as the children voxel center
        outPos.index_put_({at::indexing::Slice(), outIdx, 0}, prevPos.index({at::indexing::Slice(), parentIdx, 0}) - scale + scale * 0.95f * xIdx + absoluteScale / 2.0f);
        outPos.index_put_({at::indexing::Slice(), outIdx, 1}, prevPos.index({at::indexing::Slice(), parentIdx, 1}) - scale + scale * 0.95f * yIdx + absoluteScale / 2.0f);
        outPos.index_put_({at::indexing::Slice(), outIdx, 2}, prevPos.index({at::indexing::Slice(), parentIdx, 2}) - scale + scale * 0.95f * zIdx + absoluteScale / 2.0f);
    }
    return outPos;
}

void dumpToBin(std::string outPath,
               at::Tensor &level0,
               at::Tensor &level1,
               at::Tensor &level2,
               const int octreeRootNum){
    std::ofstream file(outPath, std::ios::out | std::ios::binary);
    const float octreeRootNumFloat = static_cast<float>(octreeRootNum);
    file.write(reinterpret_cast<const char*>(&octreeRootNumFloat), sizeof(float));

    std::vector<at::Tensor> level0_out;
    std::vector<at::Tensor> level1_out;
    std::vector<at::Tensor> level2_out;

    for (int i = 0; i < octreeRootNum; i++){
        if (torch::sum(torch::abs(level0.index({i, at::indexing::Slice(at::indexing::None, 7 * 7 *7)}))).item().toFloat() > 0.1){
            break;
        }
        at::Tensor l0_out_with_pc = torch::zeros({level0.size(1) + 2});
        l0_out_with_pc[0] = i; // Children index
        l0_out_with_pc[1] = -1;
        l0_out_with_pc.index_put_({at::indexing::Slice(2, at::indexing::None)}, level0.index({i}));
        level0_out.push_back(l0_out_with_pc);
    }

    for (int i = 0; i < octreeRootNum * 8; i++){
        if (torch::sum(torch::abs(level1.index({i, at::indexing::Slice(at::indexing::None, 5 * 5 * 5)}))).item().toFloat() > 0.1){
            break;
        }
        at::Tensor l1_out_with_pc = torch::zeros({level1.size(1) + 2});
        l1_out_with_pc[0] = i % 8 ; // Children index
        l1_out_with_pc[1] = i / 8;
        l1_out_with_pc.index_put_({at::indexing::Slice(2, at::indexing::None)}, level1.index({i}));
        level1_out.push_back(l1_out_with_pc);
    }

    for (int i = 0; i < octreeRootNum * 8 * 8; i++){
        if (torch::sum(torch::abs(level2.index({i, at::indexing::Slice(at::indexing::None, 5 * 5 * 5)}))).item().toFloat() > 0.1){
            break;
        }
        at::Tensor l2_out_with_pc = torch::zeros({level2.size(1) + 2});
        l2_out_with_pc[0] = i % 8 ; // Children index
        l2_out_with_pc[1] = i / 8;
        l2_out_with_pc.index_put_({at::indexing::Slice(2, at::indexing::None)}, level2.index({i}));
        level2_out.push_back(l2_out_with_pc);
    }

    float length0f = static_cast<float>(level0_out.size());
    float length1f = static_cast<float>(level1_out.size());
    float length2f = static_cast<float>(level2_out.size());
    file.write(reinterpret_cast<const char*>(&length0f), sizeof(float));
    file.write(reinterpret_cast<const char*>(&length1f), sizeof(float));
    file.write(reinterpret_cast<const char*>(&length2f), sizeof(float));

    for (auto &t: level0_out){
        file.write(reinterpret_cast<const char*>(t.data_ptr<float>()), sizeof(float) * (level0.size(1) + 2));
    }
    for (auto &t: level1_out){
        file.write(reinterpret_cast<const char*>(t.data_ptr<float>()), sizeof(float) * (level1.size(1) + 2));
    }
    for (auto &t: level2_out){
        file.write(reinterpret_cast<const char*>(t.data_ptr<float>()), sizeof(float) * (level2.size(1) + 2));
    }

    file.close();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load", &readAndConstruct, "Read binary and construct tree tensors");
  m.def("max_voxel_length", &getMaximumVoxelLength,
        "Get the maximum root voxel length for normalization.");
  m.def("dump_to_bin", &dumpToBin,
        "Dump vectors to the form of binary files.");
  m.def("deduce_position_from_sample", &deducePositionFromSample,
        "Deduce position from samples.");
}