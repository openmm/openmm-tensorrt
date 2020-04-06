#include "CudaNeuralNetworkKernels.h"
#include "CudaNeuralNetworkKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace OpenMM;

CudaCalcTensorRTForceKernel::~CudaCalcTensorRTForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void CudaCalcTensorRTForceKernel::initialize(const System& system, const TensorRTForce& force, TF_Session* session, TF_Graph* graph) {

    cu.setAsCurrent();
    this->session = session;
    this->graph = graph;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Construct input tensors.

    const int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(TF_FLOAT, positionsDims, 2, numParticles*3*TF_DataTypeSize(TF_FLOAT));
    if (usePeriodic) {
        const int64_t boxVectorsDims[] = {3, 3};
        boxVectorsTensor = TF_AllocateTensor(TF_FLOAT, boxVectorsDims, 2, 9*TF_DataTypeSize(TF_FLOAT));
    }

    // Inititalize CUDA objects.
    networkForces.initialize(cu, 3*numParticles, TF_DataTypeSize(TF_FLOAT), "networkForces");

    // Create kernles
    auto module = cu.createModule(CudaNeuralNetworkKernelSources::neuralNetworkForce);
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

    double energy = 0.0;
    if (includeEnergy)
        energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];

    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }

    return energy;
}
