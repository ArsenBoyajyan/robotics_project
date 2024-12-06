import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_coffee_edges(image_path):
    """
    使用Sobel算子检测咖啡图像的边缘
    
    参数:
    image_path (str): 输入图像的路径
    
    返回:
    tuple: 原始图像, 灰度图像, x方向边缘, y方向边缘, 综合边缘图
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X方向边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))
    
    # Sobel Y方向边缘检测
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(255 * sobel_y / np.max(sobel_y))
    
    # 综合边缘检测（组合X和Y方向）
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    return image, gray, sobel_x, sobel_y, sobel_combined

def process_coffee_photos(photo_folder):
    """
    处理photo文件夹中的所有图像并显示边缘检测结果
    
    参数:
    photo_folder (str): 包含咖啡图像的文件夹路径
    """
    # 获取文件夹中的所有图像
    image_files = [f for f in os.listdir(photo_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for idx, image_file in enumerate(image_files):
        # 完整图像路径
        image_path = os.path.join(photo_folder, image_file)
        
        # 执行边缘检测
        orig, gray, sobel_x, sobel_y, sobel_combined = detect_coffee_edges(image_path)
        
        # 创建matplotlib图形
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 创建5个子图
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Figure')
        axes[0].axis('off')
        
        # 灰度图像
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Gray Figure')
        axes[1].axis('off')
        
        # X方向Sobel边缘
        axes[2].imshow(sobel_x, cmap='gray')
        axes[2].set_title('X-edge')
        axes[2].axis('off')
        
        # Y方向Sobel边缘
        axes[3].imshow(sobel_y, cmap='gray')
        axes[3].set_title('Y-edge')
        axes[3].axis('off')
        
        # 综合边缘
        axes[4].imshow(sobel_combined, cmap='gray')
        axes[4].set_title('Combined-edge')
        axes[4].axis('off')
        
        # 只保存最后一个子图——"综合边缘"
        if idx == len(image_files) - 1:
            fig_combined, ax_combined = plt.subplots(figsize=(6, 6))  # 单个子图
            ax_combined.imshow(sobel_combined, cmap='gray')
            ax_combined.set_title('Combined-edge')
            ax_combined.axis('off')
            
            # 保存该子图为文件
            fig_combined.savefig("sobel_combined_result.png", bbox_inches='tight')
        
        # 显示图像
        plt.tight_layout()
        plt.show()



# 使用示例
# 请在运行前替换 'path/to/your/photo/folder' 为实际的文件夹路径
process_coffee_photos('/home/user/xiao/ME467/project/photo')