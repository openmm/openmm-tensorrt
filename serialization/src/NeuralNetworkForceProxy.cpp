#include "NeuralNetworkForceProxy.h"
#include "NeuralNetworkForce.h"
#include "openmm/serialization/SerializationNode.h"

using namespace OpenMM;

NeuralNetworkForceProxy::NeuralNetworkForceProxy() : SerializationProxy("TensorRTForce") {
}

void NeuralNetworkForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const auto& force = *reinterpret_cast<const TensorRTForce*>(object);
    node.setStringProperty("file", force.getFile());
}

void* NeuralNetworkForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    auto force = new TensorRTForce(node.getStringProperty("file"));
    return force;
}
