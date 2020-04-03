#include "NeuralNetworkForceProxy.h"
#include "NeuralNetworkForce.h"
#include "openmm/serialization/SerializationNode.h"

using namespace NNPlugin;

OpenMM::NeuralNetworkForceProxy::NeuralNetworkForceProxy() : SerializationProxy("NeuralNetworkForce") {
}

void OpenMM::NeuralNetworkForceProxy::serialize(const void* object, OpenMM::SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const NeuralNetworkForce& force = *reinterpret_cast<const NeuralNetworkForce*>(object);
    node.setStringProperty("file", force.getFile());
}

void* OpenMM::NeuralNetworkForceProxy::deserialize(const OpenMM::SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMM::OpenMMException("Unsupported version number");
    NeuralNetworkForce* force = new NeuralNetworkForce(node.getStringProperty("file"));
    return force;
}
