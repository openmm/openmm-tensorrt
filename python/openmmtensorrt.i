%module openmmtensorrt

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "TensorRTForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace OpenMM {

class TensorRTForce : public Force {
public:
    TensorRTForce(const std::string& file);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * Add methods for casting a Force to a TensorRTForce.
    */
    %extend {
        static TensorRTForce& cast(Force& force) {
            return dynamic_cast<OpenMM::TensorRTForce&>(force);
        }

        static bool isinstance(Force& force) {
            return (dynamic_cast<OpenMM::TensorRTForce*>(&force) != NULL);
        }
    }
};

}
