#ifndef OPENMM_TENSORRT_FORCE_PROXY_H_
#define OPENMM_TENSORRT_FORCE_PROXY_H_

#include "internal/windowsExportNN.h"
#include "openmm/serialization/SerializationProxy.h"

namespace OpenMM {

/**
 * This is a proxy for serializing TensorRTForce objects.
 */

class OPENMM_EXPORT_NN TensorRTForceProxy : public SerializationProxy {
public:
    TensorRTForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_TENSORRT_FORCE_PROXY_H_*/
