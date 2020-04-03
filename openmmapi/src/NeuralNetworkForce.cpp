#include "NeuralNetworkForce.h"
#include "internal/NeuralNetworkForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace NNPlugin;

NeuralNetworkForce::NeuralNetworkForce(const std::string& file) : file(file), usePeriodic(false) {
    std::ifstream graphFile(file);
    graphProto = std::string((std::istreambuf_iterator<char>(graphFile)), std::istreambuf_iterator<char>());
}

OpenMM::ForceImpl* NeuralNetworkForce::createImpl() const {
    return new NeuralNetworkForceImpl(*this);
}
