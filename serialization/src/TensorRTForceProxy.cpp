#include "TensorRTForceProxy.h"
#include "TensorRTForce.h"
#include "openmm/serialization/SerializationNode.h"

using namespace OpenMM;

TensorRTForceProxy::TensorRTForceProxy() : SerializationProxy("TensorRTForce") {
}

void TensorRTForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const auto& force = *reinterpret_cast<const TensorRTForce*>(object);
    node.setStringProperty("file", force.getFile());
}

void* TensorRTForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    auto force = new TensorRTForce(node.getStringProperty("file"));
    return force;
}
