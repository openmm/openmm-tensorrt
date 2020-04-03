#include <exception>

#include "CudaNeuralNetworkKernelFactory.h"
#include "CudaNeuralNetworkKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace NNPlugin;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        auto& platform = OpenMM::Platform::getPlatformByName("CUDA");
        auto factory = new OpenMM::CudaNeuralNetworkKernelFactory();
        platform.registerKernelFactory(CalcNeuralNetworkForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerNeuralNetworkCudaKernelFactories() {
    try {
        OpenMM::Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        OpenMM::Platform::registerPlatform(new OpenMM::CudaPlatform());
    }
    registerKernelFactories();
}

OpenMM::KernelImpl* OpenMM::CudaNeuralNetworkKernelFactory::createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const {
    auto& cu = *static_cast<OpenMM::CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcNeuralNetworkForceKernel::Name())
        return new CudaCalcNeuralNetworkForceKernel(name, platform, cu);
    throw OpenMM::OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
