#include "TensorRTForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

extern "C" void registerTensorRTSerializationProxies();

void testSerialization() {
    // Create a Force.

    TensorRTForce force("graph.pb", "dummy");

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<TensorRTForce>(&force, "Force", buffer);
    TensorRTForce* copy = XmlSerializer::deserialize<TensorRTForce>(buffer);

    // Compare the two forces to see if they are identical.

    TensorRTForce& force2 = *copy;
    ASSERT_EQUAL(force.getFile(), force2.getFile());
}

int main() {
    try {
        registerTensorRTSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
