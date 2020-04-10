#include "TensorRTForce.h"
#include "internal/TensorRTForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>
#include <sstream>

using namespace OpenMM;

TensorRTForce::TensorRTForce(const std::string& file, const std::string& file2) : file(file), file2(file2), usePeriodic(false) {
    std::ifstream graphFile(file);
    graphProto = std::string((std::istreambuf_iterator<char>(graphFile)), std::istreambuf_iterator<char>());

    std::stringstream stream;
    stream << std::ifstream(file2, std::ifstream::binary).rdbuf();
    serializedGraph = stream.str();
}

ForceImpl* TensorRTForce::createImpl() const {
    return new TensorRTForceImpl(*this);
}
