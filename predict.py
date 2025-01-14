import os
import torch
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import mymodel

# 加载模型（这里假设你已经有训练好的U-Net模型）
def load_model(model_path):
    model =mymodel.LF_Unet()
    model.load_state_dict(torch.load(model_path))  # 加载预训练权重
    model.eval()  # 设置模型为推理模式
    return model


# 图像预处理函数
def preprocess_image(image, input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image


# 将预测的显著图保存为图片
def save_saliency_map(saliency_map, output_path):
    # 将预测结果转换为图像格式
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())  # 归一化
    saliency_map = (saliency_map * 255).astype(np.uint8)

    Image.fromarray(saliency_map).save(output_path)


# 主函数：预测显著图
def predict_saliency(input_folder, output_folder, model, input_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # 图像预处理
        input_image = preprocess_image(image, input_size)

        # 将图像输入模型进行预测
        with torch.no_grad():
            saliency_map = model(input_image)
            saliency_map = F.interpolate(saliency_map, size=image.size[::-1], mode='bilinear', align_corners=False)
            saliency_map = torch.sigmoid(saliency_map)  # 使用sigmoid获得显著图

        # 保存预测的显著图
        output_path = os.path.join(output_folder, image_name)
        save_saliency_map(saliency_map, output_path)

        print(f"Saved saliency map for {image_name} at {output_path}")


# 示例使用
if __name__ == '__main__':
    input_folder = '../dataset/test_array_reshape'  # 输入图片文件夹
    output_folder = './checkpoints/predicted_TEST5'  # 输出显著图的保存文件夹
    model_path = './checkpoints/best_model_epoch_5.pth'  # 模型文件路径
    input_size = (640*5, 640*5)  # 输入图片的尺寸，可以根据你的模型调整

    model = load_model(model_path)
    predict_saliency(input_folder, output_folder, model, input_size)
