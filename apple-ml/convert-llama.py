import coremltools as ct

def fromPyTorch(sampleinput, tracedmodel):
    model = ct.convert(
    tracedmodel,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=sampleinput.shape)])
    return

    
fromPyTorch(None, None).save("out.mlpackage")