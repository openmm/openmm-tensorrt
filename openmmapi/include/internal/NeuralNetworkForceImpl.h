#ifndef OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_
#define OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_

#include "NeuralNetworkForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>

namespace NNPlugin {

class System;

/**
 * This is the internal implementation of NeuralNetworkForce.
 */

class OPENMM_EXPORT_NN NeuralNetworkForceImpl : public OpenMM::ForceImpl {
public:
    NeuralNetworkForceImpl(const NeuralNetworkForce& owner);
    ~NeuralNetworkForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const NeuralNetworkForce& getOwner() const { return owner; }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {}
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() { return {}; }
    std::vector<std::string> getKernelNames();
private:
    const NeuralNetworkForce& owner;
    OpenMM::Kernel kernel;
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;
};

} // namespace NNPlugin

#endif /*OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_*/
