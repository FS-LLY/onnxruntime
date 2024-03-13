import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
 
model_fp32 = "./VGG16_infer.onnx"
model_quant = "./VGG16_SVHN_single_int8.onnx"
 
# 加载FP32模型
onnx_model = onnx.load(model_fp32)
 
quantize_dynamic(model_fp32,model_quant,weight_type=QuantType.QUInt8)
 