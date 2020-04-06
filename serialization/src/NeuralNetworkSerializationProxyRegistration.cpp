#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "TensorRTForce.h"
#include "NeuralNetworkForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_NN void registerNeuralNetworkSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerNeuralNetworkSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerNeuralNetworkSerializationProxies();
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT_NN void registerNeuralNetworkSerializationProxies() {
    SerializationProxy::registerProxy(typeid(TensorRTForce), new NeuralNetworkForceProxy());
}
