#include "CudaTensorRTKernels.h"
#include "CudaTensorRTKernelSources.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

void CudaCalcTensorRTForceKernel::initialize(const System& system, const TensorRTForce& force, Engine& engine) {

    cu.setAsCurrent();

    const int numParticles = system.getNumParticles();
    usePeriodic = force.usesPeriodicBoundaryConditions();

    // Create TensorRT execution context
    // TODO fix the destructor
    const auto destructor = [](ExecutionContext* e) { /* e->destroy(); */ };
    execution = {engine.createExecutionContext(), destructor};

    // Initialize CUDA arrays
    graphPositions.initialize<float>(cu, 3*numParticles, "graphPosition");
    if (usePeriodic)
        graphVectors.initialize<float>(cu, 9, "graphVectors");
    graphEnergy.initialize<float>(cu, 1, "graphEnergy");
    graphForces.initialize<float>(cu, 3*numParticles, "graphForces2");

    // Create biding for the graph execution
    static_assert(sizeof(CUdeviceptr) == sizeof(void*));
    bindings.push_back(reinterpret_cast<void*>(graphPositions.getDevicePointer()));
    if (usePeriodic)
        bindings.push_back(reinterpret_cast<void*>(graphVectors.getDevicePointer()));
    bindings.push_back(reinterpret_cast<void*>(graphEnergy.getDevicePointer()));
    bindings.push_back(reinterpret_cast<void*>(graphForces.getDevicePointer()));

    // Create kernels
    auto module = cu.createModule(CudaTensorRTKernelSources::TensorRTForce);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcTensorRTForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    std::vector<Vec3> positions;
    context.getPositions(positions);
    std::vector<float> positionsArray;
    for (const auto& atom: positions) {
        positionsArray.push_back(atom[0]);
        positionsArray.push_back(atom[1]);
        positionsArray.push_back(atom[2]);
    }
    graphPositions.upload(positionsArray);

    if (usePeriodic) {
        std::vector<float> vectorArray;
        Vec3 vectors[3];
        cu.getPeriodicBoxVectors(vectors[0], vectors[1], vectors[2]);
        for (int i = 0; i < 3; i++) {
            vectorArray.push_back(vectors[i][0]);
            vectorArray.push_back(vectors[i][1]);
            vectorArray.push_back(vectors[i][2]);
        }
        graphVectors.upload(vectorArray);
    }

    // Execute the graph
    execution->executeV2(bindings.data());

    double energy = 0.0;
    if (includeEnergy) {
        std::vector<float> energyArray;
        graphEnergy.download(energyArray);
        energy = energyArray[0];
    }

    if (includeForces) {
        int numAtoms = cu.getNumAtoms();
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&graphForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numAtoms, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numAtoms);
    }

    return energy;
}
