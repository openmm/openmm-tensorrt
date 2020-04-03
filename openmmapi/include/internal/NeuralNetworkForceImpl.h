#ifndef OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_
#define OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_

#include "NeuralNetworkForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <c_api.h>
#include <utility>
#include <set>
#include <string>

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
    const NeuralNetworkForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
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
