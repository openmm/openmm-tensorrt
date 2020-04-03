#include "NeuralNetworkForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerNeuralNetworkSerializationProxies();

void testSerialization() {
    // Create a Force.

    NeuralNetworkForce force("graph.pb");

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<NeuralNetworkForce>(&force, "Force", buffer);
    NeuralNetworkForce* copy = XmlSerializer::deserialize<NeuralNetworkForce>(buffer);

    // Compare the two forces to see if they are identical.

    NeuralNetworkForce& force2 = *copy;
    ASSERT_EQUAL(force.getFile(), force2.getFile());
}

int main() {
    try {
        registerNeuralNetworkSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
