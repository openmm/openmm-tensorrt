#include "TensorRTForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OpenMM;

extern "C" void registerTensorRTSerializationProxies();

void testSerialization() {
    // Create a Force.

    TensorRTForce force("graphs/aperiodic.trt");

    // Serialize and then deserialize it.

    std::stringstream buffer;
    XmlSerializer::serialize<TensorRTForce>(&force, "Force", buffer);
    TensorRTForce* copy = XmlSerializer::deserialize<TensorRTForce>(buffer);

    // Compare the two forces to see if they are identical.

    TensorRTForce& force2 = *copy;
    ASSERT_EQUAL(force.getSerializedGraph(), force2.getSerializedGraph());
}

int main() {
    try {
        registerTensorRTSerializationProxies();
        testSerialization();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
