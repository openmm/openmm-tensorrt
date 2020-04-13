#ifndef OPENMM_TENSORRT_KERNELS_H_
#define OPENMM_TENSORRT_KERNELS_H_

#include "TensorRTForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <NvInfer.h>
#include <string>

namespace OpenMM {

/**
 * This kernel is invoked by TensorRTForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcTesorRTForceKernel : public KernelImpl {
public:
    using Engine = nvinfer1::ICudaEngine;
    static std::string Name() {
        return "CalcTensorRTForce";
    }
    CalcTesorRTForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {}
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the TenorRTForce this kernel will be used for
     */
    virtual void initialize(const System& system, const TensorRTForce& force, Engine& engine) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace OpenMM

#endif /*OPENMM_TENSORRT_KERNELS_H_*/
