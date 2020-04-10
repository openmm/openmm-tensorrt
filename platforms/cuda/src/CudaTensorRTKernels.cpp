#include "CudaTensorRTKernels.h"
#include "CudaTensorRTKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace OpenMM;

CudaCalcTensorRTForceKernel::~CudaCalcTensorRTForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void CudaCalcTensorRTForceKernel::initialize(const System& system, const TensorRTForce& force, TF_Session* session, TF_Graph* graph, Engine& engine) {

    cu.setAsCurrent();
    this->session = session;
    this->graph = graph;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Create TensorRT execution context
    // TODO fix the destructor
    const auto destructor = [](ExecutionContext* e) { /* e->destroy(); */ };
    execution = {engine.createExecutionContext(), destructor};

    // Construct input tensors.

    const int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(TF_FLOAT, positionsDims, 2, numParticles*3*TF_DataTypeSize(TF_FLOAT));
    if (usePeriodic) {
        const int64_t boxVectorsDims[] = {3, 3};
        boxVectorsTensor = TF_AllocateTensor(TF_FLOAT, boxVectorsDims, 2, 9*TF_DataTypeSize(TF_FLOAT));
    }

    // Inititalize CUDA objects.
    graphForces.initialize(cu, 3*numParticles, TF_DataTypeSize(TF_FLOAT), "graphForces");

    graphPositions.initialize<float>(cu, 3*numParticles, "graphPosition");
    if (usePeriodic)
        graphVectors.initialize<float>(cu, 9, "graphVectors");
    graphEnergy.initialize<float>(cu, 1, "graphEnergy");
    graphForces2.initialize<float>(cu, 3*numParticles, "graphForces2");

    static_assert(sizeof(CUdeviceptr) == sizeof(void*));

    bindings.push_back(reinterpret_cast<void*>(graphPositions.getDevicePointer()));
    if (usePeriodic)
        bindings.push_back(reinterpret_cast<void*>(graphVectors.getDevicePointer()));
    bindings.push_back(reinterpret_cast<void*>(graphEnergy.getDevicePointer()));
    bindings.push_back(reinterpret_cast<void*>(graphForces2.getDevicePointer()));

    // Create kernles
    auto module = cu.createModule(CudaTensorRTKernelSources::TensorRTForce);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcTensorRTForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    std::vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    auto positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
    for (int i = 0; i < numParticles; i++) {
        positions[3*i] = pos[i][0];
        positions[3*i+1] = pos[i][1];
        positions[3*i+2] = pos[i][2];
    }

    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        auto boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = box[i][j];
    }

    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> inputTensors;
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
        inputTensors.push_back(boxVectorsTensor);
    }

    std::vector<TF_Output> outputs;
    int forceOutputIndex = 0;
    if (includeEnergy)
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
    }
    std::vector<TF_Tensor*> outputTensors(outputs.size());

    auto status = TF_NewStatus();
    TF_SessionRun(session, NULL, &inputs[0], &inputTensors[0], inputs.size(),
                  &outputs[0], &outputTensors[0], outputs.size(),
                  NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(std::string("Error running TensorFlow session: ")+TF_Message(status));
    TF_DeleteStatus(status);

    std::vector<float> positions2;
    for (const auto& p: pos) {
        positions2.push_back(p[0]);
        positions2.push_back(p[1]);
        positions2.push_back(p[2]);
    }
    graphPositions.upload(positions2);

    if (usePeriodic) {
        std::vector<float> vectors;
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++) {
            vectors.push_back(box[i][0]);
            vectors.push_back(box[i][1]);
            vectors.push_back(box[i][2]);
        }
        graphVectors.upload(vectors);
    }

    execution->executeV2(bindings.data());

    double energy = 0.0;
    if (includeEnergy)
        energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];

    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        graphForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&graphForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }

    return energy;
}
