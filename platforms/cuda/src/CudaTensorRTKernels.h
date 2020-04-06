#ifndef OPENMM_CUDA_TENSORRT_KERNELS_H_
#define OPENMM_CUDA_TENSORRT_KERNELS_H_

#include "TensorRTKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace OpenMM {

/**
 * This kernel is invoked by TensorRTForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcTensorRTForceKernel : public CalcTesorRTForceKernel {
public:
    CudaCalcTensorRTForceKernel(std::string name, const Platform& platform, CudaContext& cu) :
            CalcTesorRTForceKernel(name, platform), hasInitializedKernel(false), cu(cu),
            positionsTensor(NULL), boxVectorsTensor(NULL) {
    }
    ~CudaCalcTensorRTForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the TensorRTForce this kernel will be used for
     * @param session        the TensorFlow session in which to do calculations
     * @param graph          the TensorFlow graph to use for computing forces and energy
     */
    void initialize(const System& system, const TensorRTForce& force, TF_Session* session, TF_Graph* graph);
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
    TF_Session* session;
    TF_Graph* graph;
    TF_Tensor* positionsTensor;
    TF_Tensor* boxVectorsTensor;
    bool usePeriodic;
    CudaArray graphForces;
    CUfunction addForcesKernel;
};

} // namespace OpenMM

#endif /*OPENMM_CUDA_TENSORRT_KERNELS_H_*/
