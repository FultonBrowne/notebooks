from transformers import AutoTokenizer, AutoModelForCausalLM
import coremltools as ct
import torch

def fromPyTorch(sampleinput, tracedmodel):
    model = ct.convert(
    tracedmodel,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=sampleinput.shape)])
    return model
# LOAD THE MODEL
model = AutoModelForCausalLM.from_pretrained("gpt2")
# TRACE THE MODEL
model.eval()
example_input = torch.randint(10000, (768,1))
print(model)
traced_model = torch.jit.trace(model, example_input)
# EXPORT THE MODEL
out = traced_model(example_input, traced_model)    
fromPyTorch(None, None).save("out.mlpackage")