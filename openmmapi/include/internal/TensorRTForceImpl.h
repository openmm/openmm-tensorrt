#ifndef OPENMM_TENSORRT_FORCE_IMPL_H_
#define OPENMM_TENSORRT_FORCE_IMPL_H_

#include "TensorRTForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <tensorflow/c/c_api.h>
#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace OpenMM {

class Logger : public nvinfer1:: ILogger {
    void log(Severity severity, const char* msg) override {
        std::cout << msg << std::endl;
    }
};

/**
 * This is the internal implementation of TensorRTForce.
 */

class OPENMM_EXPORT_NN TensorRTForceImpl : public ForceImpl {
public:
    TensorRTForceImpl(const TensorRTForce& owner);
    ~TensorRTForceImpl();
    void initialize(ContextImpl& context);
    const TensorRTForce& getOwner() const { return owner; }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {}
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() { return {}; }
    std::vector<std::string> getKernelNames();
private:
    const TensorRTForce& owner;
    Kernel kernel;
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;
    Logger logger;
    using Runtime = nvinfer1::IRuntime;
    using Engine = nvinfer1::ICudaEngine;
    std::shared_ptr<Engine> engine;
    std::shared_ptr<Runtime> runtime;
};

} // namespace OpenMM

#endif /*OPENMM_TENSORRT_FORCE_IMPL_H_*/
