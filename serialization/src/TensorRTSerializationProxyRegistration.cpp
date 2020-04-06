#ifdef WIN32
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "TensorRTForce.h"
#include "TensorRTForceProxy.h"
#include "openmm/serialization/SerializationProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_NN void registerTensorRTSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerTensorRTSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerTensorRTSerializationProxies();
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT_NN void registerTensorRTSerializationProxies() {
    SerializationProxy::registerProxy(typeid(TensorRTForce), new TensorRTForceProxy());
}
