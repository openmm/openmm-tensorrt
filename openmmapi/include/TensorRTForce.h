#ifndef OPENMM_TENSORRT_FORCE_H_
#define OPENMM_TENSORRT_FORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include "internal/windowsExportNN.h"

namespace OpenMM {

/**
 * This class implements forces that are defined by an user-supplied computational graph.
 * It uses TensorRT library to perform the computations. */

class OPENMM_EXPORT_NN TensorRTForce : public Force {
public:
    /**
     * Create a TensorRTForce. The force is defined by a TensorRT graph saved
     * to a binary file.
     *
     * @param file   the path to the file containing the graph
     */
    TensorRTForce(const std::string& file, const std::string& file2);
    /**
     * Get the path to the file containing the graph.
     */
    const std::string& getFile() const { return file; }
    /**
     * Get the content of the protocol buffer defining the graph.
     */
    const std::string& getSerializedGraph() const { return serializedGraph; }
    /**
     * Set whether this force makes use of periodic boundary conditions.  If this is set
     * to true, the TensorFlow graph must include a 3x3 tensor called "boxvectors", which
     * is set to the current periodic box vectors.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic) { usePeriodic = periodic; }
    /**
     * Get whether this force makes use of periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const { return usePeriodic; }
protected:
    ForceImpl* createImpl() const;
private:
    std::string file;
    std::string serializedGraph;
    bool usePeriodic;
};

} // namespace OpenMM

#endif /*OPENMM_TENSORRT_FORCE_H_*/
