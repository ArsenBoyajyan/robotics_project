import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_sobel_image(image_path):
    """
    读取并显示 sobel_combined_result 图像，调用拉花艺术生成函数
    
    参数:
    image_path (str): 图像的文件路径
    """
    # 读取sobel_combined_result.png图像
    sobel_combined = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 确保图像加载成功
    if sobel_combined is None:
        print("图像加载失败，请检查文件路径和文件名！")
    else:
        # 可视化边缘检测结果
        plt.imshow(sobel_combined, cmap='gray')
        plt.title('边缘检测结果')
        plt.axis('off')
        plt.show()
        
# 示例使用
if __name__ == "__main__":
    # 替换为你的 sobel_combined_result.png 图像路径
    image_path = "/home/user/xiao/ME467/project/sobel_combined_result.png"
    load_and_visualize_sobel_image(image_path)

       
