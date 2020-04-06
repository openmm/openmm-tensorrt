#include <exception>

#include "CudaTensorRTKernelFactory.h"
#include "CudaTensorRTKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        auto& platform = Platform::getPlatformByName("CUDA");
        auto factory = new CudaTensorRTKernelFactory();
        platform.registerKernelFactory(CalcTesorRTForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerTensorRTCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaTensorRTKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    auto& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcTesorRTForceKernel::Name())
        return new CudaCalcTensorRTForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
