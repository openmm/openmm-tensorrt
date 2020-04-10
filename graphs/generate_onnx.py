import torch

class Graph(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, positions, vectors=None):

        if vectors is not None:
            vectors = torch.flatten(vectors)
            vectors = torch.cat((vectors[0:1], vectors[4:5], vectors[8:9]))
            positions = positions - torch.floor(positions / vectors) * vectors

        energy = torch.sum(positions**2)
        forces = -2 * positions

        return energy, forces

if __name__ == '__main__':

    graph = Graph()
    positions = torch.zeros((10, 3), dtype=torch.float32)
    vectors = torch.zeros((3, 3), dtype=torch.float32)

    torch.onnx.export(graph, (positions,), 'aperiodic.onnx', verbose=True, opset_version=9)
    torch.onnx.export(graph, (positions, vectors), 'periodic.onnx', verbose=True, opset_version=9)