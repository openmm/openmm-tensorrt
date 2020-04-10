#include "TensorRTForceProxy.h"
#include "TensorRTForce.h"
#include "openmm/serialization/SerializationNode.h"

using namespace OpenMM;

void TensorRTForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const auto& force = *reinterpret_cast<const TensorRTForce*>(object);
    node.setStringProperty("serializedGraph", force.serializedGraph);
    node.setBoolProperty("usePeriodic", force.usePeriodic);
}

void* TensorRTForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    const auto& serializedGraph = node.getStringProperty("serializedGraph");
    bool usePeriodic = node.getBoolProperty("usePeriodic");
    return new TensorRTForce(serializedGraph, usePeriodic);
}
