#include "TensorRTForce.h"
#include "internal/TensorRTForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <sstream>

using namespace OpenMM;

TensorRTForce::TensorRTForce(const std::string& file): file(file), usePeriodic(false) {

    // Read the serialized graph from a file
    std::stringstream stream;
    stream << std::ifstream(file, std::ifstream::binary).rdbuf();
    serializedGraph = stream.str();
}

ForceImpl* TensorRTForce::createImpl() const {
    return new TensorRTForceImpl(*this);
}
