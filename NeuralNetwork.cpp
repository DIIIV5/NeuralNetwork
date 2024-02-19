// NeuralNetwork.cpp : Defines the entry point for the application.

#include "NeuralNetwork.h"

const int SEED = 133;

//Implemetation of NeuralNetwork
NeuralNetwork::NeuralNetwork(int NumberOfLayers, int SizeOfLayers, int SizeOfInput, int SizeOfOutput) {
    this->NumberOfLayers = NumberOfLayers;
    this->SizeOfLayers = SizeOfLayers;
    this->SizeOfInput = SizeOfInput;
    this->SizeOfOutput = SizeOfOutput;

    LearningRate = 0.01;

    CreateWeights(NumberOfLayers, SizeOfLayers, SizeOfInput, SizeOfOutput);
}

void NeuralNetwork::CreateWeights(int& NumberOfLayers, int& SizeOfLayers, int& SizeOfInput, int& SizeOfOutput) {
    //Weights of the first layer
    Weights.InputWeights =  torch::randn({SizeOfLayers, SizeOfInput}, torch::TensorOptions().dtype(DATA_TYPE).device(Device));

    //Weights of the last layer
    Weights.OutputWeights = torch::randn({ SizeOfOutput, SizeOfLayers }, torch::TensorOptions().dtype(DATA_TYPE).device(Device));

    //Weights of the hidden
    Weights.HiddenLayerWeights = torch::randn({ NumberOfLayers - 2, SizeOfLayers, SizeOfLayers }, torch::TensorOptions().dtype(DATA_TYPE).device(Device));
}

torch::Tensor NeuralNetwork::ForwardPass(torch::Tensor Input) {
    torch::Tensor Logits;

    try {
        Input = Input.flatten();
        Input = Input.to(DATA_TYPE).to(Device);

        // Forward pass
        // Input layer
        torch::Tensor Output = torch::matmul(Weights.InputWeights, Input);
        Output = torch::relu(Output) + 1;
        // Hidden layers
        for (int i = 0; i < NumberOfLayers - 2; ++i) {
            Output = torch::matmul(Weights.HiddenLayerWeights[i], Output).to(Device);
            Output = torch::relu(Output).to(Device) + 1;
        }
        // Output Layer
        Logits = torch::matmul(Weights.OutputWeights, Output);
    }
    catch (const torch::Error& e) {
        // Handle specific torch::Error exceptions
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        // Handle all other std::exception based exceptions
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...) {
        // Handle any other type of exception
        std::cerr << "An unknown exception occurred." << std::endl;
    }

    return Logits;
}

//returns the average loss from the pass
float NeuralNetwork::Pass(const DataStruct& Data, const float& CurrentLearningRate) {
    float Loss;

    try {
        // Get data tensor ready
        torch::Tensor Input = Data.Input.to(Device);
        torch::Tensor Target = Data.Target.to(Device);
        Input = Input.flatten();
        Input = Input.to(DATA_TYPE);

        std::vector<torch::Tensor> ZTensors; //stores the raw outputs from the forward pass
        std::vector<torch::Tensor> ATensors; //stores the activations from the forward pass

        torch::nn::CrossEntropyLoss LossFunction;

        // Forward pass
        // Input layer
        torch::Tensor Output = torch::matmul(Weights.InputWeights, Input);
        ZTensors.push_back(Output);
        Output = torch::relu(Output) + 1;
        ATensors.push_back(Output);
        // Hidden layers
        for (int i = 0; i < NumberOfLayers - 2; ++i) {
            Output = torch::matmul(Weights.HiddenLayerWeights[i], Output).to(Device);
            ZTensors.push_back(Output);
            Output = torch::relu(Output).to(Device) + 1;
            ATensors.push_back(Output);
        }
        // Output Layer
        Output = torch::matmul(Weights.OutputWeights, Output);
        ZTensors.push_back(Output);
        Output = torch::softmax(Output, /*dim=*/0).to(Device);

        // calculate error, predictions minus the expected value
        torch::Tensor Error = Output - Target;
        Loss = LossFunction(ZTensors[NumberOfLayers - 1], Target).item<float>();
        ATensors.push_back(Output);

        // Backpropagate starting with the output layer

        torch::Tensor Gradient = torch::matmul(Error.unsqueeze(1), ZTensors[NumberOfLayers - 2].unsqueeze(0));

        // Save weights for the next layer
        torch::Tensor PreviousWeights = Weights.OutputWeights;
        //Update the output layers weights
        Weights.OutputWeights -= CurrentLearningRate * Gradient;

        // Backpropagate through hidden layers
        for (int i = NumberOfLayers - 2; i > 0; --i) {
            Error = torch::matmul(ZTensors[i + 1].unsqueeze(1).transpose(0,1), PreviousWeights);
            torch::Tensor ReLUDerivative = (Error > 0).to(DATA_TYPE);
            Gradient = torch::matmul(ATensors[i - 1].unsqueeze(1), ReLUDerivative);
            PreviousWeights = Weights.HiddenLayerWeights[i - 1];
            Weights.HiddenLayerWeights[i - 1] -= CurrentLearningRate * Gradient;
        }

        // Backpropagate through the input layer
        Error = torch::matmul(ZTensors[0].unsqueeze(1).transpose(0, 1), PreviousWeights);
        torch::Tensor ReLUDerivative = (Error > 0).to(DATA_TYPE);
        Gradient = torch::matmul(ReLUDerivative.transpose(0,1), Input.unsqueeze(1).transpose(0, 1));
        Weights.InputWeights -= CurrentLearningRate * Gradient;
    }

    catch (const torch::Error& e) {
        // Handle specific torch::Error exceptions
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        // Handle all other std::exception based exceptions
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
    catch (...) {
        // Handle any other type of exception
        std::cerr << "An unknown exception occurred." << std::endl;
    }

    return Loss;
}

void NeuralNetwork::Train(float LearningRate, unsigned int Epochs = 1) {
    this->LearningRate = LearningRate;
    float CurrentLearningRate = LearningRate;
    float DLearningRate = (CurrentLearningRate * 0.9f) / (Epochs + 1.0f);

    std::cout << "Neural Network Training\n" << "Learning Rate: " << CurrentLearningRate << "\nEpochs: " << Epochs << std::endl;
    std::cout << "Change in learning rate per Epoch: " << DLearningRate << '\n' << "Validation:" << '\n';;
    Validate();

    std::cout << "\n\nBeginning Training\n" << std::endl;

    for (unsigned int Epoch = 1; Epoch <= Epochs; ++Epoch) {
        std::cout << "Epoch " << Epoch << " of " << Epochs << std::endl;
        std::cout << "Learning Rate: " << CurrentLearningRate << '\n';

        //shuffle data
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        std::shuffle(TrainingData.begin(), TrainingData.end(), std::default_random_engine(seed));

        float TotalLoss = 0.0f;
        for (DataStruct Data : TrainingData) {
            try {
                float Loss = Pass(Data, CurrentLearningRate);
                TotalLoss += Loss / TrainingData.size();
            }
            catch (const torch::Error& e) {
                // Handle specific torch::Error exceptions
                std::cerr << "Error: " << e.what() << std::endl;
            }
            catch (const std::exception& e) {
                // Handle all other std::exception based exceptions
                std::cerr << "Standard exception: " << e.what() << std::endl;
            }
            catch (...) {
                // Handle any other type of exception
                std::cerr << "An unknown exception occurred." << std::endl;
            }
        }
        std::cout << "Training Loss: " << TotalLoss << std::endl;
        std::cout << "Validation" << std::endl;
        Validate();
        CurrentLearningRate = CurrentLearningRate - DLearningRate;
    }
    std::cout << "\nTraining Complete" << std::endl;
}

void NeuralNetwork::Validate() {
    float TotalLoss = 0.0f;
    unsigned int Correct = 0;
    for (DataStruct Data : ValidationData) {
        try {
            torch::Tensor Prediction = ForwardPass(Data.Input).to(Device);

            torch::nn::CrossEntropyLoss LossFunction;
            Data.Target = Data.Target.to(Device);
            if (torch::equal(torch::softmax(Prediction, /*dim=*/0), Data.Target)) { ++Correct; }
            float Loss = LossFunction(Prediction, Data.Target).item<float>();
            TotalLoss += Loss / ValidationData.size();
        }
        catch (const torch::Error& e) {
            // Handle specific torch::Error exceptions
            std::cerr << "Error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            // Handle all other std::exception based exceptions
            std::cerr << "Standard exception: " << e.what() << std::endl;
        }
        catch (...) {
            // Handle any other type of exception
            std::cerr << "An unknown exception occurred." << std::endl;
        }
    }

    std::cout << Correct << "/" << ValidationData.size() << " correct\n";
    std::cout << "Accuracy: " << (Correct * 1.0f) / ValidationData.size() <<  '\n';
    std::cout << "Validation Loss: " << TotalLoss << "\n\n";
}

torch::Tensor NeuralNetwork::GetPrediction(torch::Tensor Output) {
    return torch::Tensor();
}

void NeuralNetwork::InsertTrainingData(const std::vector<DataStruct>& Data ) {
    TrainingData = Data;
}

void NeuralNetwork::InsertValidationData(const std::vector<DataStruct>& Data) {
    ValidationData = Data;
}

void NeuralNetwork::InsertTestData(const std::vector<DataStruct>& Data) {
    TestData = Data;
}

bool NeuralNetwork::Save(std::string Path) {
    std::cout << "Save Network" << std::endl;
    return false;
}

bool NeuralNetwork::Load(std::string Path) {
    std::cout << "Load Network" << std::endl;
    return false;
}

float NeuralNetwork::CalculateLoss() {
    return -1.0f;
}

int NeuralNetwork::GetNumberOfLayers() {
    return NumberOfLayers;
}

int NeuralNetwork::GetSizeOfLayers() {
    return SizeOfLayers;
}

bool NeuralNetwork::ToCUDA() {
    if (torch::cuda::is_available()) {
        Device = torch::Device(torch::kCUDA);
        Weights.InputWeights = Weights.InputWeights.to(Device);
        Weights.HiddenLayerWeights = Weights.HiddenLayerWeights.to(Device);
        Weights.OutputWeights = Weights.OutputWeights.to(Device);
        OnCUDA = true;
        return true;
    }
    else {
        return false;
    }
}

void NeuralNetwork::Print() const {
    std::cout << "Input:\n\n" << Weights.InputWeights << "\nHiddenLayers:\n\n" << Weights.HiddenLayerWeights << "\nOutput:\n\n" << Weights.OutputWeights << std::endl;
}

void NeuralNetwork::PrintShape() const {
    std::cout << "Input:\n\n" << Weights.InputWeights.sizes() << "\nHiddenLayers:\n\n" << Weights.HiddenLayerWeights.sizes() << "\nOutput:\n\n" << Weights.OutputWeights.sizes() << std::endl;
}

// Helper functions
bool IsImage(const std::string& FilePath) {
    cv::Mat image = cv::imread(FilePath, cv::IMREAD_UNCHANGED);
    return !image.empty();
}

std::vector<torch::Tensor> ToTensors(std::vector<cv::Mat>& Images) {
    std::vector<torch::Tensor> Tensors;
    for (auto img : Images) {
        torch::Tensor tensor = torch::from_blob(img.data, { img.rows, img.cols }, torch::kUInt8);
        Tensors.push_back(tensor);
    }

    return Tensors;
}

struct DataClass {
    fs::path DirectoryPath;
    std::vector<torch::Tensor> DataTensors;
    torch::Tensor TargetTensor;
};

int main(int argc, char* argv[]) {
    std::cout << "Hello World" << std::endl;
    if (argc >= 6) {
        std::cout << "NumberOfLayers: " << argv[1] << '\n';
        std::cout << "SizeOfLayers: " << argv[2] << '\n';
        std::cout << "SizeOfInput: " << argv[3] << '\n';
        std::cout << "NumberOfEpochs: " << argv[4] << '\n';
        std::cout << "LearningRate: " << argv[5] << '\n';

        std::stringstream convert{ argv[1] }; // set up a stringstream variable named convert, initialized with the input from argv[1]

        if (!(convert >> NumberOfLayers)) { // do the conversion
            std::cerr << "Could not convert arg to NumberOfLayers";
            return 1;
        }
        convert = std::stringstream{ argv[2] };
        if (!(convert >> SizeOfLayers)) { // do the conversion
            std::cerr << "Could not convert arg to SizeOfLayers";
            return 1;
        }
        convert = std::stringstream{ argv[3] };
        if (!(convert >> SizeOfInput)) { // do the conversion
            std::cerr << "Could not convert arg to SizeOfInput";
            return 1;
        }
        convert = std::stringstream{ argv[4] };
        if (!(convert >> NumberOfEpochs)) { // do the conversion
            std::cerr << "Could not convert arg to NumberOfEpochs";
            return 1;
        }
        convert = std::stringstream{ argv[5] };
        if (!(convert >> LearningRate)) { // do the conversion
            std::cerr << "Could not convert arg to LearningRate";
            return 1;
        }
    }

    std::vector<DataStruct> TrainingData;
    std::vector<DataStruct> TestData;
    std::vector<DataStruct> ValidationData;

    std::vector<DataClass> DataClasses;

    for (int i = 0; i < MathSymbol::Size; ++i) {
        std::cout << "Getting Images:" << std::endl;
        DataClass DataClass;

        DataClass.DirectoryPath = PROJECT_ROOT + MathSymbol::Path.at(MathSymbol::SYMBOL_LIST[i]);

        std::cout << DataClass.DirectoryPath << std::endl;
        torch::Tensor InitializerTensor = torch::zeros(MathSymbol::Size, DATA_TYPE);
        InitializerTensor[i] = 1;
        DataClass.TargetTensor = InitializerTensor;

        std::vector<cv::Mat> Images;
        for (const auto& entry : fs::directory_iterator(DataClass.DirectoryPath)) {
            std::string FilePath = entry.path().string();
            cv::Mat img = cv::imread(FilePath, cv::IMREAD_GRAYSCALE);
            if (!img.empty()) {
                cv::resize(img, img, cv::Size(SizeOfInput, SizeOfInput));
                Images.push_back(img);
            }
        }

        DataClass.DataTensors = ToTensors(Images);

        DataClasses.push_back(DataClass);
    }

    for (int i = 0; i < DataClasses.size(); ++i) {
        DataClass DataClass = DataClasses[i];
        for (int j = 0; j < DataClass.DataTensors.size(); ++j) {
            if (j % 200 == 0) { TestData.push_back({ DataClass.DataTensors[i], DataClass.TargetTensor }); }
            if (j % 100 == 7) { ValidationData.push_back({ DataClass.DataTensors[i], DataClass.TargetTensor }); }
            else { TrainingData.push_back({ DataClass.DataTensors[i], DataClass.TargetTensor }); }
        }
    }

    std::cout << "Creating Network" << std::endl;
    NeuralNetwork Network(NumberOfLayers, SizeOfLayers, SizeOfInput * SizeOfInput, MathSymbol::Size);
    Network.ToCUDA();

    Network.InsertTestData(TestData);
    Network.InsertTrainingData(TrainingData);
    Network.InsertValidationData(ValidationData);
    
    std::cout << "train" << std::endl;
    Network.Train(LearningRate, NumberOfEpochs);

    return 0;
}