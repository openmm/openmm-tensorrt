#ifndef OPENMM_NEURAL_NETWORK_FORCE_PROXY_H_
#define OPENMM_NEURAL_NETWORK_FORCE_PROXY_H_

#include "internal/windowsExportNN.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

/**
 * This is a proxy for serializing NeuralNetworkForce objects.
 */

class OPENMM_EXPORT_NN NeuralNetworkForceProxy : public SerializationProxy {
public:
    NeuralNetworkForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_NEURAL_NETWORK_FORCE_PROXY_H_*/
