import tensorrt as trt

def build(onnx_file, trt_file):

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 32
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(open(onnx_file, 'rb').read())
    engine = builder.build_cuda_engine(network)

    for i in range(engine.num_bindings):
        print(i, engine.binding_is_input(i),
                 engine.get_binding_dtype(i),
                 engine.get_binding_format(i),
                 engine.get_binding_name(i), 
                 engine.get_binding_shape(i),
                 engine.get_binding_vectorized_dim(i),
                 engine.get_location(i))

    with open(trt_file, 'wb') as f:
        f.write(engine.serialize())

if __name__ == '__main__':

    build('aperiodic.onnx', 'aperiodic.trt')
    build('periodic.onnx', 'periodic.trt')