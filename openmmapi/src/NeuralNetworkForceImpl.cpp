#include "internal/NeuralNetworkForceImpl.h"
#include "NeuralNetworkKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace NNPlugin;

NeuralNetworkForceImpl::NeuralNetworkForceImpl(const NeuralNetworkForce& owner) : owner(owner), graph(NULL), session(NULL), status(TF_NewStatus()) {}

NeuralNetworkForceImpl::~NeuralNetworkForceImpl() {
    if (session != NULL) {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
    }
    if (graph != NULL)
        TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void NeuralNetworkForceImpl::initialize(OpenMM::ContextImpl& context) {
    // Load the graph from the file.

    const auto& graphProto = owner.getGraphProto();
    auto buffer = TF_NewBufferFromString(graphProto.c_str(), graphProto.size());
    graph = TF_NewGraph();
    auto importOptions = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, importOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMM::OpenMMException(std::string("Error loading TensorFlow graph: ")+TF_Message(status));
    TF_DeleteImportGraphDefOptions(importOptions);
    TF_DeleteBuffer(buffer);

    // Check that the graph contains all the expected elements and that their types
    // are supported.

    TF_Output positions = {TF_GraphOperationByName(graph, "positions"), 0};
    if (positions.oper == NULL)
        throw OpenMM::OpenMMException("NeuralNetworkForce: the graph does not have a 'positions' input");
    const auto positionsType = TF_OperationOutputType(positions);
    if (positionsType != TF_FLOAT && positionsType != TF_DOUBLE)
        throw OpenMM::OpenMMException("NeuralNetworkForce: 'positions' must have type float32 or float64");
    auto boxType = TF_FLOAT;
    if (owner.usesPeriodicBoundaryConditions()) {
        TF_Output boxvectors = {TF_GraphOperationByName(graph, "boxvectors"), 0};
        if (boxvectors.oper == NULL)
            throw OpenMM::OpenMMException("NeuralNetworkForce: the graph does not have a 'boxvectors' input");
        boxType = TF_OperationOutputType(boxvectors);
        if (boxType != TF_FLOAT && boxType != TF_DOUBLE)
            throw OpenMM::OpenMMException("NeuralNetworkForce: 'boxvectors' must have type float32 or float64");
    }
    TF_Output energy = {TF_GraphOperationByName(graph, "energy"), 0};
    if (energy.oper == NULL)
        throw OpenMM::OpenMMException("NeuralNetworkForce: the graph does not have an 'energy' output");
    const auto energyType = TF_OperationOutputType(energy);
    if (energyType != TF_FLOAT && energyType != TF_DOUBLE)
        throw OpenMM::OpenMMException("NeuralNetworkForce: 'energy' must have type float32 or float64");
    TF_Output forces = {TF_GraphOperationByName(graph, "forces"), 0};
    if (forces.oper == NULL)
        throw OpenMM::OpenMMException("NeuralNetworkForce: the graph does not have a 'forces' output");
    const auto forcesType = TF_OperationOutputType(forces);
    if (forcesType != TF_FLOAT && forcesType != TF_DOUBLE)
        throw OpenMM::OpenMMException("NeuralNetworkForce: 'forces' must have type float32 or float64");

    // Create the TensorFlow Session.

    auto sessionOptions = TF_NewSessionOptions();
    session = TF_NewSession(graph, sessionOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMM::OpenMMException(std::string("Error creating TensorFlow session: ")+TF_Message(status));
    TF_DeleteSessionOptions(sessionOptions);

    // Create the kernel.

    kernel = context.getPlatform().createKernel(CalcNeuralNetworkForceKernel::Name(), context);
    kernel.getAs<CalcNeuralNetworkForceKernel>().initialize(context.getSystem(), owner, session, graph,
            positionsType, boxType, energyType, forcesType);
}

double NeuralNetworkForceImpl::calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcNeuralNetworkForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> NeuralNetworkForceImpl::getKernelNames() {
    return { CalcNeuralNetworkForceKernel::Name() };
}
