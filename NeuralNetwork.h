// NeuralNetwork.h : Defines the entry point for the application.
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>

// External Libraries
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Datasets.h"

#define Convert(arg) std::stringstream convert{ arg }

const auto DATA_TYPE = torch::kFloat32;

namespace fs = std::filesystem;
namespace MathSymbol = HandWrittenMathSymbols;

std::string PROJECT_ROOT = "";
int NumberOfLayers = 5;
int SizeOfLayers = 1024;
int SizeOfInput = 32;
int NumberOfEpochs = 20;
float LearningRate = 0.000000001;

struct DataStruct {
    torch::Tensor Input; //Input tensor
    torch::Tensor Target; //Expected output
};

// A Neural Network, uses a float32 tensor
class NeuralNetwork {

public:

    // Creates a network with an input layer, output layer and NumberOfLayers - 2 hidden layers
    NeuralNetwork(int NumberOfLayers, int SizeOfLayers, int SizeOfInput, int SizeOfOutput);

    void CreateWeights(int& NumberOfLayers, int& SizeOfLayers, int& SizeOfInput, int& SizeOfOutput);

    float Pass(const DataStruct& Data, const float& CurrentLearningRate);

    void Train(float LearningRate, unsigned int Epochs);

    void Validate();

    torch::Tensor ForwardPass(torch::Tensor Input);

    torch::Tensor GetPrediction(torch::Tensor Output);

    void InsertTrainingData(const std::vector<DataStruct>& Data);

    void InsertValidationData(const std::vector<DataStruct>& Data);

    void InsertTestData(const std::vector<DataStruct>& Data);

    void Test();

    bool ToCUDA();

    void Print() const;

    void PrintShape() const;

    //TODO
    bool Save(std::string Path="./checkpoints");

    //TODO
    bool Load(std::string Path="./checkpoints");

    float CalculateLoss();

    int GetNumberOfLayers();

    int GetSizeOfLayers();

    std::vector<DataStruct> TrainingData;

    std::vector<DataStruct> ValidationData;

    std::vector<DataStruct> TestData;

private:

    struct WeightsStruct {
        torch::Tensor InputWeights;
        torch::Tensor HiddenLayerWeights;
        torch::Tensor OutputWeights;
    } Weights;

    int NumberOfLayers;

    int SizeOfLayers;

    int SizeOfInput;

    int SizeOfOutput;

    float LearningRate;

    bool OnCUDA = false;

    torch::Device Device = torch::Device(torch::kCPU);

    std::vector<torch::Tensor> DataTensors;
};