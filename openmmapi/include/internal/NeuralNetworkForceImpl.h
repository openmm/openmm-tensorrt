#ifndef OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_
#define OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_

#include "NeuralNetworkForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This is the internal implementation of NeuralNetworkForce.
 */

class OPENMM_EXPORT_NN NeuralNetworkForceImpl : public ForceImpl {
public:
    NeuralNetworkForceImpl(const NeuralNetworkForce& owner);
    ~NeuralNetworkForceImpl();
    void initialize(ContextImpl& context);
    const NeuralNetworkForce& getOwner() const { return owner; }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {}
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() { return {}; }
    std::vector<std::string> getKernelNames();
private:
    const NeuralNetworkForce& owner;
    Kernel kernel;
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;
};

} // namespace OpenMM

#endif /*OPENMM_NEURAL_NETWORK_FORCE_IMPL_H_*/
