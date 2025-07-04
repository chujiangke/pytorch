import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

# 字符集 - 应与训练时使用的字符集一致
char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 1. 加载ONNX模型
def load_onnx_model(model_path):
    # 尝试使用GPU加速，如果不可用则回退到CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return onnxruntime.InferenceSession(model_path, providers=providers)

# 2. 图像预处理 - 适配单通道灰度图
def preprocess_image(image_path):
    # 与训练时相同的预处理流程
    transform = transforms.Compose([
        transforms.Resize((60, 160)),       # 调整到模型需要的尺寸
        transforms.Grayscale(num_output_channels=1),  # 确保转换为单通道
        transforms.ToTensor(),              # 转为张量
        transforms.Normalize(mean=[0.5], std=[0.5])   # 单通道的归一化
    ])
    
    image = Image.open(image_path).convert('L')  # 确保加载为灰度图
    return transform(image).unsqueeze(0).numpy()  # 添加batch维度并转为numpy

# 3. 执行推理
def predict_captcha(onnx_session, image_path):
    # 预处理图像
    input_data = preprocess_image(image_path)
    
    # 运行推理
    outputs = onnx_session.run(
        None, 
        {'input': input_data.astype(np.float32)}
    )[0]
    
    # 处理输出 (假设输出为4个字符，每个字符36类)
    # 输出形状应为 (1, 144) -> 4个字符 * 36个类别
    outputs = outputs.reshape(4, 36)  # 调整为 (4, 36)
    captcha = ""
    
    for char_logits in outputs:
        char_index = np.argmax(char_logits)
        captcha += char_set[char_index]
    
    return captcha

# 4. 使用示例
if __name__ == "__main__":
    # 加载模型
    onnx_session = load_onnx_model("saved_models/captcha_model.onnx")
    
    # 测试图像
    test_images = [
        "test_captcha1.png",
        "test_captcha2.png",
        "test_captcha3.png"
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"警告: 测试图像 {img_path} 不存在，跳过")
            continue
            
        prediction = predict_captcha(onnx_session, img_path)
        print(f"图像: {img_path} -> 预测结果: {prediction}")
