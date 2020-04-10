#include "internal/TensorRTForceImpl.h"
#include "TensorRTKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

TensorRTForceImpl::TensorRTForceImpl(const TensorRTForce& owner) : owner(owner), graph(NULL), session(NULL), status(TF_NewStatus()) {

    const auto destructor = [](Runtime* r) { r->destroy(); };
    runtime = {nvinfer1::createInferRuntime(logger), destructor};
}

TensorRTForceImpl::~TensorRTForceImpl() {
    if (session != NULL) {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
    }
    if (graph != NULL)
        TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void TensorRTForceImpl::initialize(ContextImpl& context) {
    // Load the graph from the file.

    const auto& graphProto = owner.getGraphProto();
    auto buffer = TF_NewBufferFromString(graphProto.c_str(), graphProto.size());
    graph = TF_NewGraph();
    auto importOptions = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, importOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(std::string("Error loading TensorFlow graph: ")+TF_Message(status));
    TF_DeleteImportGraphDefOptions(importOptions);
    TF_DeleteBuffer(buffer);

    // Deserialize TensorRT graph
    const auto& graph2 = owner.getSerializedGraph();
    const auto destructor = [](Engine* e) { e->destroy(); };
    engine = {runtime->deserializeCudaEngine(graph2.data(), graph2.size()), destructor};

    // Check that the graph contains all the expected elements and that their types
    // are supported.

    TF_Output positions = {TF_GraphOperationByName(graph, "positions"), 0};
    if (positions.oper == NULL)
        throw OpenMMException("TensorRTForce: the graph does not have a 'positions' input");
    if (TF_OperationOutputType(positions) != TF_FLOAT)
        throw OpenMMException("TensorRTForce: 'positions' must have type float32");

    if (owner.usesPeriodicBoundaryConditions()) {
        TF_Output boxvectors = {TF_GraphOperationByName(graph, "boxvectors"), 0};
        if (boxvectors.oper == NULL)
            throw OpenMMException("TensorRTForce: the graph does not have a 'boxvectors' input");
        if (TF_OperationOutputType(boxvectors) != TF_FLOAT)
            throw OpenMMException("TensorRTForce: 'boxvectors' must have type float32");
    }

    TF_Output energy = {TF_GraphOperationByName(graph, "energy"), 0};
    if (energy.oper == NULL)
        throw OpenMMException("TensorRTForce: the graph does not have an 'energy' output");
    if (TF_OperationOutputType(energy) != TF_FLOAT)
        throw OpenMMException("TensorRTForce: 'energy' must have type float32");

    TF_Output forces = {TF_GraphOperationByName(graph, "forces"), 0};
    if (forces.oper == NULL)
        throw OpenMMException("TensorRTForce: the graph does not have a 'forces' output");
    if (TF_OperationOutputType(forces) != TF_FLOAT)
        throw OpenMMException("TensorRTForce: 'forces' must have type float32");

    // Create the TensorFlow Session.

    auto sessionOptions = TF_NewSessionOptions();
    session = TF_NewSession(graph, sessionOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(std::string("Error creating TensorFlow session: ")+TF_Message(status));
    TF_DeleteSessionOptions(sessionOptions);

    // Validate TesorRT graph
    const auto periodic = owner.usesPeriodicBoundaryConditions();
    const auto numBindings = engine->getNbBindings();

    if (periodic && numBindings != 4)
        throw OpenMMException("TensorRTForce: graph mush have 4 bindings");

    if (!periodic && numBindings != 3)
        throw OpenMMException("TensorRTForce: graph must have 3 bindings");

    // TODO complete validation

    // Create the kernel.

    kernel = context.getPlatform().createKernel(CalcTesorRTForceKernel::Name(), context);
    kernel.getAs<CalcTesorRTForceKernel>().initialize(context.getSystem(), owner, session, graph, *engine);
}

double TensorRTForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcTesorRTForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> TensorRTForceImpl::getKernelNames() {
    return { CalcTesorRTForceKernel::Name() };
}
