#include "NeuralNetworkForce.h"
#include "internal/NeuralNetworkForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

NeuralNetworkForce::NeuralNetworkForce(const std::string& file) : file(file), usePeriodic(false) {
    ifstream graphFile(file);
    graphProto = string((istreambuf_iterator<char>(graphFile)), istreambuf_iterator<char>());
}

const string& NeuralNetworkForce::getFile() const {
    return file;
}

const string& NeuralNetworkForce::getGraphProto() const {
    return graphProto;
}

ForceImpl* NeuralNetworkForce::createImpl() const {
    return new NeuralNetworkForceImpl(*this);
}

void NeuralNetworkForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool NeuralNetworkForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
