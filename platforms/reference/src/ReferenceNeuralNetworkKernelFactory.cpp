#include "ReferenceNeuralNetworkKernelFactory.h"
#include "ReferenceNeuralNetworkKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace NNPlugin;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < OpenMM::Platform::getNumPlatforms(); i++) {
        auto& platform = OpenMM::Platform::getPlatform(i);
        if (dynamic_cast<OpenMM::ReferencePlatform*>(&platform) != NULL) {
            auto factory = new OpenMM::ReferenceNeuralNetworkKernelFactory();
            platform.registerKernelFactory(CalcNeuralNetworkForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerNeuralNetworkReferenceKernelFactories() {
    registerKernelFactories();
}

OpenMM::KernelImpl* OpenMM::ReferenceNeuralNetworkKernelFactory::createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const {
    auto& data = *static_cast<OpenMM::ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcNeuralNetworkForceKernel::Name())
        return new ReferenceCalcNeuralNetworkForceKernel(name, platform);
    throw OpenMM::OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
