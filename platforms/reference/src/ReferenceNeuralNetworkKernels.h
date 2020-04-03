#ifndef REFERENCE_NEURAL_NETWORK_KERNELS_H_
#define REFERENCE_NEURAL_NETWORK_KERNELS_H_

#include "NeuralNetworkKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace NNPlugin {

/**
 * This kernel is invoked by NeuralNetworkForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcNeuralNetworkForceKernel : public CalcNeuralNetworkForceKernel {
public:
    ReferenceCalcNeuralNetworkForceKernel(std::string name, const OpenMM::Platform& platform) : CalcNeuralNetworkForceKernel(name, platform),
            positionsTensor(NULL), boxVectorsTensor(NULL) {
    }
    ~ReferenceCalcNeuralNetworkForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the NeuralNetworkForce this kernel will be used for
     * @param session        the TensorFlow session in which to do calculations
     * @param graph          the TensorFlow graph to use for computing forces and energy
     * @param positionsType  the data type of the "positions" tensor
     * @param boxType        the data type of the "boxvectors" tensor
     * @param energyType     the data type of the "energy" tensor
     * @param forcesType     the data type of the "forces" tensor
     */
    void initialize(const OpenMM::System& system, const NeuralNetworkForce& force, TF_Session* session, TF_Graph* graph,
                    TF_DataType positionsType, TF_DataType boxType, TF_DataType energyType, TF_DataType forcesType);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    TF_Session* session;
    TF_Graph* graph;
    TF_Tensor* positionsTensor;
    TF_Tensor* boxVectorsTensor;
    TF_DataType positionsType, boxType, energyType, forcesType;
    std::vector<float> positions, boxVectors;
    bool usePeriodic;
};

} // namespace NNPlugin

#endif /*REFERENCE_NEURAL_NETWORK_KERNELS_H_*/
