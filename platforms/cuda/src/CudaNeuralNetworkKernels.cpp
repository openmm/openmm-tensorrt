#include "CudaNeuralNetworkKernels.h"
#include "CudaNeuralNetworkKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace NNPlugin;

CudaCalcNeuralNetworkForceKernel::~CudaCalcNeuralNetworkForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void CudaCalcNeuralNetworkForceKernel::initialize(const OpenMM::System& system, const NeuralNetworkForce& force, TF_Session* session, TF_Graph* graph,
            TF_DataType positionsType, TF_DataType boxType, TF_DataType energyType, TF_DataType forcesType) {
    cu.setAsCurrent();
    this->session = session;
    this->graph = graph;
    this->positionsType = positionsType;
    this->boxType = boxType;
    this->energyType = energyType;
    this->forcesType = forcesType;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Construct input tensors.

    const int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(positionsType, positionsDims, 2, numParticles*3*TF_DataTypeSize(positionsType));
    if (usePeriodic) {
        const int64_t boxVectorsDims[] = {3, 3};
        boxVectorsTensor = TF_AllocateTensor(boxType, boxVectorsDims, 2, 9*TF_DataTypeSize(boxType));
    }

    // Inititalize CUDA objects.

    networkForces.initialize(cu, 3*numParticles, TF_DataTypeSize(forcesType), "networkForces");
    std::map<std::string, std::string> defines;
    if (forcesType == TF_FLOAT)
        defines["FORCES_TYPE"] = "float";
    else
        defines["FORCES_TYPE"] = "double";
    CUmodule module = cu.createModule(CudaNeuralNetworkKernelSources::neuralNetworkForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcNeuralNetworkForceKernel::execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) {
    std::vector<OpenMM::Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    if (positionsType == TF_FLOAT) {
        auto positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    else {
        auto positions = reinterpret_cast<double*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    if (usePeriodic) {
        OpenMM::Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (boxType == TF_FLOAT) {
            auto boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
        else {
            auto boxVectors = reinterpret_cast<double*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
    }
    std::vector<TF_Output> inputs, outputs;
    int forceOutputIndex = 0;
    if (includeEnergy)
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
    }
    std::vector<TF_Tensor*> inputTensors, outputTensors(outputs.size());
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
        inputTensors.push_back(boxVectorsTensor);
    }
    auto status = TF_NewStatus();
    TF_SessionRun(session, NULL, &inputs[0], &inputTensors[0], inputs.size(),
                  &outputs[0], &outputTensors[0], outputs.size(),
                  NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMM::OpenMMException(std::string("Error running TensorFlow session: ")+TF_Message(status));
    TF_DeleteStatus(status);
    double energy = 0.0;
    if (includeEnergy) {
        if (energyType == TF_FLOAT)
            energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];
        else
            energy = reinterpret_cast<double*>(TF_TensorData(outputTensors[0]))[0];
    }
    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
