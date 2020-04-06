#include "TensorRTForce.h"
#include "internal/TensorRTForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace OpenMM;

TensorRTForce::TensorRTForce(const std::string& file) : file(file), usePeriodic(false) {
    std::ifstream graphFile(file);
    graphProto = std::string((std::istreambuf_iterator<char>(graphFile)), std::istreambuf_iterator<char>());
}

ForceImpl* TensorRTForce::createImpl() const {
    return new TensorRTForceImpl(*this);
}
