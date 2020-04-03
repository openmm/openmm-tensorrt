#include <exception>

#include "CudaNeuralNetworkKernelFactory.h"
#include "CudaNeuralNetworkKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        int argc = 0;
        vector<char**> argv = {NULL};
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaNeuralNetworkKernelFactory* factory = new CudaNeuralNetworkKernelFactory();
        platform.registerKernelFactory(CalcNeuralNetworkForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerNeuralNetworkCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaNeuralNetworkKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcNeuralNetworkForceKernel::Name())
        return new CudaCalcNeuralNetworkForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
