# NeuralNetwork

A project that creates aand trains a neural network. It needs polishing but it works

Installation:

    • Download and place the libtorch and opencv libraries in the libraries folder, or modify the cmake file
    • Change the PROJECT_ROOT string in the .h file the full path to your project root
    • Build the project with cmake

Usage:

    • NeuralNetwork takes exactly 5 command line arguments - NumberOfLayers SizeOfLayers SizeOfInput NumberOfEpochs LearningRate
        ex. on Windows: NeuralNetwork.exe 5 128 64 10 0.0000002
    • You can also run it with no arguments with the default values