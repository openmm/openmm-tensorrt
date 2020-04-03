#include "ReferenceNeuralNetworkKernelFactory.h"
#include "ReferenceNeuralNetworkKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    int argc = 0;
    vector<char**> argv = {NULL};
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceNeuralNetworkKernelFactory* factory = new ReferenceNeuralNetworkKernelFactory();
            platform.registerKernelFactory(CalcNeuralNetworkForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerNeuralNetworkReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceNeuralNetworkKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcNeuralNetworkForceKernel::Name())
        return new ReferenceCalcNeuralNetworkForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
