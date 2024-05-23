//
// Dingdong Yang, 04/01/2023
//

#include <regex>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

#include <torch/extension.h>

namespace fs = std::filesystem;

struct Quaternion {
    float w, x, y, z;
};

// Function to compute the cross product of two vectors
void crossProduct(const float v1[3], const float v2[3], float result[3]) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v1[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// Function to convert a rotation matrix to a quaternion
Quaternion rotationMatrixToQuaternion(const float *mat) {
    // Compute the third row using cross product
    float r3[3];
    crossProduct(mat, mat + 3, r3);

    // Assemble the full rotation matrix
    float fullMat[9] = {
        mat[0], mat[1], mat[2],
        mat[3], mat[4], mat[5],
        r3[0], r3[1], r3[2]
    };

    // Compute the trace of the matrix
    float trace = fullMat[0] + fullMat[4] + fullMat[8];

    Quaternion q;
    if (trace > 0) {
        float s = 0.5f / std::sqrt(trace + 1.0f);
        q.w = 0.25f / s;
        q.x = (fullMat[7] - fullMat[5]) * s;
        q.y = (fullMat[2] - fullMat[6]) * s;
        q.z = (fullMat[3] - fullMat[1]) * s;
    } else {
        if (fullMat[0] > fullMat[4] && fullMat[0] > fullMat[8]) {
            float s = 2.0f * std::sqrt(1.0f + fullMat[0] - fullMat[4] - fullMat[8]);
            q.w = (fullMat[7] - fullMat[5]) / s;
            q.x = 0.25f * s;
            q.y = (fullMat[1] + fullMat[3]) / s;
            q.z = (fullMat[2] + fullMat[6]) / s;
        } else if (fullMat[4] > fullMat[8]) {
            float s = 2.0f * std::sqrt(1.0f + fullMat[4] - fullMat[0] - fullMat[8]);
            q.w = (fullMat[2] - fullMat[6]) / s;
            q.x = (fullMat[1] + fullMat[3]) / s;
            q.y = 0.25f * s;
            q.z = (fullMat[5] + fullMat[7]) / s;
        } else {
            float s = 2.0f * std::sqrt(1.0f + fullMat[8] - fullMat[0] - fullMat[4]);
            q.w = (fullMat[3] - fullMat[1]) / s;
            q.x = (fullMat[2] + fullMat[6]) / s;
            q.y = (fullMat[5] + fullMat[7]) / s;
            q.z = 0.25f * s;
        }
    }

    return q;
}

// Function to convert a quaternion to a rotation matrix
void quaternionToRotationMatrix(const Quaternion& q, float mat[9]) {
    float w = q.w, x = q.x, y = q.y, z = q.z;

    mat[0] = 1 - 2 * y * y - 2 * z * z;
    mat[1] = 2 * x * y - 2 * z * w;
    mat[2] = 2 * x * z + 2 * y * w;

    mat[3] = 2 * x * y + 2 * z * w;
    mat[4] = 1 - 2 * x * x - 2 * z * z;
    mat[5] = 2 * y * z - 2 * x * w;

    mat[6] = 2 * x * z - 2 * y * w;
    mat[7] = 2 * y * z + 2 * x * w;
    mat[8] = 1 - 2 * x * x - 2 * y * y;
}

void normalizeQuaternion(Quaternion& q) {
    float rnorm = (std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) + 1e-6f);
    q.w *= rnorm;
    q.x *= rnorm;
    q.y *= rnorm;
    q.z *= rnorm;
}

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
        auto dataPtr = data;
        file.read(reinterpret_cast<char*>(data), sizeof(float) * unitLength);
        int parentIdx = rootFlag ? 0 : static_cast<int>(data[1]);
        int currentIdx = static_cast<int>(data[0]);
        at::Tensor nodeVec = at::from_blob(data + 2, {unitLength - 2}).clone();

        // Load grid values         
        out.index_put_({parentIdx * 8 + currentIdx, at::indexing::Slice(at::indexing::None, 7 * 7 * 7)},
                       nodeVec.index({at::indexing::Slice(at::indexing::None, 7 * 7 * 7)}));

        // Do the angular encoding conversion
        data = data + 7 * 7 * 7 + 2;
        Quaternion q = rotationMatrixToQuaternion(data);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7}, q.w);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 1}, q.x);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 2}, q.y);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 3}, q.z);

        // Copy the rest
        out.index_put_({parentIdx * 8 + currentIdx, at::indexing::Slice(7 * 7 * 7 + 4, at::indexing::None)},
                       nodeVec.index({at::indexing::Slice(7 * 7 * 7 + 6, at::indexing::None)}));

        delete[] dataPtr;
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
        auto dataPtr = data;
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
        //out.index_put_({outIdx, at::indexing::Slice()}, nodeVec);

        // Load grid values         
        out.index_put_({parentIdx * 8 + currentIdx, at::indexing::Slice(at::indexing::None, 5 * 5 * 5)},
                       nodeVec.index({at::indexing::Slice(at::indexing::None, 5 * 5 * 5)}));

        // Do the angular encoding conversion
        data = data + 5 * 5 * 5 + 2;
        Quaternion q = rotationMatrixToQuaternion(data);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7}, q.w);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 1}, q.x);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 2}, q.y);
        out.index_put_({parentIdx * 8 + currentIdx, 7 * 7 * 7 + 3}, q.z);


        // Copy the rest
        out.index_put_({parentIdx * 8 + currentIdx, at::indexing::Slice(5 * 5 * 5 + 4, at::indexing::None)},
                       nodeVec.index({at::indexing::Slice(5 * 5 * 5 + 6, at::indexing::None)}));

        delete[] dataPtr;
    }
}

std::vector<at::Tensor> readAndConstruct(std::string inputPath,
                                         const int level0UnitLength = 361,
                                         const int level1UnitLength = 139){
    std::ifstream file(inputPath, std::ios::in | std::ios::binary);

    float octreeRootNumFloat;
    file.read(reinterpret_cast<char*>(&octreeRootNumFloat), sizeof(float));
    int octreeRootNum = static_cast<int>(octreeRootNumFloat);
    float length0f, length1f;
    file.read(reinterpret_cast<char*>(&length0f), sizeof(float));
    file.read(reinterpret_cast<char*>(&length1f), sizeof(float));
    int length0, length1;
    length0 = static_cast<int>(length0f);
    length1 = static_cast<int>(length1f);

    // The deducted 2 are the tree topology indicators (parent ID and children ID)
    at::Tensor level0Tensor = at::zeros({octreeRootNum, level0UnitLength - 4});
    at::Tensor level1Tensor = at::zeros({octreeRootNum * 8, level1UnitLength - 4});

    // Positions
    at::Tensor level1Positions = at::zeros({octreeRootNum * 8, 3});

    // Scales
    at::Tensor level1Scales = at::zeros({octreeRootNum * 8});

    loadFromFile(file, level0Tensor, length0, level0UnitLength, true);
    at::Tensor level0Positions = level0Tensor.index({at::indexing::Slice(), at::indexing::Slice(-3, at::indexing::None)}).clone();
    at::Tensor level0Scales = level0Tensor.index({at::indexing::Slice(), -7}).clone();
    loadFromFileAndAssignPos(file, level0Scales, level1Scales, level1Tensor, level0Positions, level1Positions, length1, level1UnitLength);

    file.close();

    return {level0Tensor, level1Tensor, level0Positions, level1Positions};
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

at::Tensor angularBack(at::Tensor inputVec, const int M){
    at::Tensor out = at::zeros({inputVec.numel() - 2});
    out.index_put_({at::indexing::Slice(at::indexing::None, M * M * M)},
                    inputVec.index({at::indexing::Slice(at::indexing::None, M * M * M)}));
    
    int i = M * M * M;
    float rotMat[9];
    Quaternion q;
    q.w = inputVec[i].item<float>();
    q.x = inputVec[i + 1].item<float>();
    q.y = inputVec[i + 2].item<float>();
    q.z = inputVec[i + 3].item<float>();
    normalizeQuaternion(q);
    quaternionToRotationMatrix(q, rotMat);
    out[i] = rotMat[0];
    out[i + 1] = rotMat[1];
    out[i + 2] = rotMat[2];
    out[i + 3] = rotMat[3];
    out[i + 4] = rotMat[4];
    out[i + 5] = rotMat[5];

    out.index_put_({at::indexing::Slice(M * M * M + 6, at::indexing::None)},
                    inputVec.index({at::indexing::Slice(M * M * M + 4, at::indexing::None)}));
    return out;
}

void dumpToBin(std::string outPath,
               at::Tensor &level0,
               at::Tensor &level1,
               const int octreeRootNum){
    std::ofstream file(outPath, std::ios::out | std::ios::binary);
    const float octreeRootNumFloat = static_cast<float>(octreeRootNum);
    file.write(reinterpret_cast<const char*>(&octreeRootNumFloat), sizeof(float));

    std::vector<at::Tensor> level0_out;
    std::vector<at::Tensor> level1_out;

    // TODO: Here is some bug

    for (int i = 0; i < octreeRootNum; i++){
        if (at::sum(at::abs(level0.index({i, at::indexing::Slice(-18, -10)}))).item().toFloat() < 1.0f){
            continue;
        }
        at::Tensor l0_out_with_pc = at::zeros({level0.size(1) + 4});
        l0_out_with_pc[0] = i; // Children index
        l0_out_with_pc[1] = -1;
        l0_out_with_pc.index_put_({at::indexing::Slice(2, at::indexing::None)}, angularBack(level0.index({i}), 7));
        level0_out.push_back(l0_out_with_pc);
    }

    for (int i = 0; i < octreeRootNum * 8; i++){
        if (at::sum(at::abs(level1.index({i, at::indexing::Slice(-14, -6)}))).item().toFloat() < 1.0f){
            continue;
        }
        at::Tensor l1_out_with_pc = at::zeros({level1.size(1) + 4});
        l1_out_with_pc[0] = i % 8 ; // Children index
        l1_out_with_pc[1] = i / 8;
        l1_out_with_pc.index_put_({at::indexing::Slice(2, at::indexing::None)}, angularBack(level1.index({i}), 5));
        level1_out.push_back(l1_out_with_pc);
    }

    float length0f = static_cast<float>(level0_out.size());
    float length1f = static_cast<float>(level1_out.size());
    file.write(reinterpret_cast<const char*>(&length0f), sizeof(float));
    file.write(reinterpret_cast<const char*>(&length1f), sizeof(float));

    for (auto &t: level0_out){
        file.write(reinterpret_cast<const char*>(t.data_ptr<float>()), sizeof(float) * (level0.size(1) + 4));
    }
    for (auto &t: level1_out){
        file.write(reinterpret_cast<const char*>(t.data_ptr<float>()), sizeof(float) * (level1.size(1) + 4));
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
