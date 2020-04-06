#ifndef NEURAL_NETWORK_KERNELS_H_
#define NEURAL_NETWORK_KERNELS_H_

#include "NeuralNetworkForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <tensorflow/c/c_api.h>
#include <string>

namespace OpenMM {

/**
 * This kernel is invoked by NeuralNetworkForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcTesorRTForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcTensorRTForce";
    }
    CalcTesorRTForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {}
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the NeuralNetworkForce this kernel will be used for
     * @param session        the TensorFlow session in which to do calculations
     * @param graph          the TensorFlow graph to use for computing forces and energy
     */
    virtual void initialize(const System& system, const TensorRTForce& force, TF_Session* session, TF_Graph* graph) = 0;
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

#endif /*NEURAL_NETWORK_KERNELS_H_*/
