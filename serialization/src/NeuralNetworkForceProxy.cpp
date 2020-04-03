#include "NeuralNetworkForceProxy.h"
#include "NeuralNetworkForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

NeuralNetworkForceProxy::NeuralNetworkForceProxy() : SerializationProxy("NeuralNetworkForce") {
}

void NeuralNetworkForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const NeuralNetworkForce& force = *reinterpret_cast<const NeuralNetworkForce*>(object);
    node.setStringProperty("file", force.getFile());
}

void* NeuralNetworkForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    NeuralNetworkForce* force = new NeuralNetworkForce(node.getStringProperty("file"));
    return force;
}
