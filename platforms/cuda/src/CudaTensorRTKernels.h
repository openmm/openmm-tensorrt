#ifndef OPENMM_CUDA_TENSORRT_KERNELS_H_
#define OPENMM_CUDA_TENSORRT_KERNELS_H_

#include "TensorRTKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <NvInfer.h>
#include <memory>

namespace OpenMM {

/**
 * This kernel is invoked by TensorRTForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcTensorRTForceKernel : public CalcTesorRTForceKernel {
public:
    CudaCalcTensorRTForceKernel(std::string name, const Platform& platform, CudaContext& cu) :
        CalcTesorRTForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the TensorRTForce this kernel will be used for
     */
    void initialize(const System& system, const TensorRTForce& force, Engine& engine);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    bool hasInitializedKernel;
    CudaContext& cu;
    bool usePeriodic;
    CUfunction addForcesKernel;
    using ExecutionContext = nvinfer1::IExecutionContext;
    std::shared_ptr<ExecutionContext> execution;
    CudaArray graphPositions;
    CudaArray graphVectors;
    CudaArray graphEnergy;
    CudaArray graphForces;
    std::vector<void*> bindings;
};

} // namespace OpenMM

#endif /*OPENMM_CUDA_TENSORRT_KERNELS_H_*/
