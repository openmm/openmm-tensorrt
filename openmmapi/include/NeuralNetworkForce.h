#ifndef OPENMM_NEURAL_NETWORKFORCE_H_
#define OPENMM_NEURAL_NETWORKFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include "internal/windowsExportNN.h"

namespace OpenMM {

/**
 * This class implements forces that are defined by user-supplied neural networks.
 * It uses the TensorFlow library to perform the computations. */

class OPENMM_EXPORT_NN TensorRTForce : public Force {
public:
    /**
     * Create a NeuralNetworkForce.  The network is defined by a TensorFlow graph saved
     * to a binary protocol buffer file.
     *
     * @param file   the path to the file containing the network
     */
    TensorRTForce(const std::string& file);
    /**
     * Get the path to the file containing the graph.
     */
    const std::string& getFile() const { return file; }
    /**
     * Get the content of the protocol buffer defining the graph.
     */
    const std::string& getGraphProto() const { return graphProto; }
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
    std::string file, graphProto;
    bool usePeriodic;
};

} // namespace OpenMM

#endif /*OPENMM_NEURAL_NETWORKFORCE_H_*/
