import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

class CoffeeLatteArt:
    def __init__(self, edge_image, num_frames=120):
        """
        初始化拉花动画生成器，增强牛奶扩散效果
        
        参数:
        edge_image (numpy.ndarray): 边缘检测后的圆形图像
        num_frames (int): 动画帧数
        """
        # 找到圆形区域
        self.mask = self._find_circular_region(edge_image)
        
        # 初始化画布
        self.canvas = np.zeros_like(edge_image, dtype=np.float32)
        self.canvas[self.mask] = 0.1  # 初始底色
        
        # 动画参数
        self.num_frames = num_frames
        self.width = edge_image.shape[1]
        self.height = edge_image.shape[0]
        
        # 扩散参数
        self.milk_history = []
    
    def _find_circular_region(self, edge_image):
        """
        识别图像中的圆形区域
        
        参数:
        edge_image (numpy.ndarray): 边缘检测后的图像
        
        返回:
        numpy.ndarray: 布尔类型的圆形区域蒙版
        """
        # 使用霍夫圆变换检测圆形
        circles = cv2.HoughCircles(
            edge_image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100, 
            param1=50, 
            param2=30, 
            minRadius=100, 
            maxRadius=400
        )
        
        # 创建蒙版
        mask = np.zeros_like(edge_image, dtype=bool)
        
        if circles is not None:
            # 取第一个检测到的圆
            circle = circles[0, 0]
            center_x, center_y, radius = map(int, circle)
            
            # 创建圆形蒙版
            Y, X = np.ogrid[:edge_image.shape[0], :edge_image.shape[1]]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
            mask = dist_from_center <= radius
        else:
            # 如果没有检测到圆，创建默认的圆形蒙版
            height, width = edge_image.shape
            Y, X = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 2 - 10
            dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
            mask = dist_from_center <= radius
        
        return mask
    
    def generate_latte_art(self):
        """
        生成拉花动画帧，增强牛奶扩散效果
        
        返回:
        list: 动画帧列表
        """
        frames = []
        
        for i in range(self.num_frames):
            # 模拟拉花过程
            self._add_milk_stream(i)
            
            # 物理扩散模拟
            self._simulate_milk_spread()
            
            # 高斯模糊创建柔和效果
            frame = gaussian_filter(self.canvas.copy(), sigma=1.5)
            
            # 归一化到0-1范围
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            
            frames.append(frame)
        
        return frames
    
    def _add_milk_stream(self, frame_num):
        """
        模拟牛奶倾倒的动态过程，增加扩散复杂性
        
        参数:
        frame_num (int): 当前帧数
        """
        # 定义牛奶流动路径
        center_x = self.width // 2
        amplitude = self.width // 8
        frequency = 0.2
        
        # 计算牛奶流动的x坐标，增加随机性
        milk_x = int(center_x + amplitude * np.sin(frequency * frame_num) + 
                     np.random.randint(-10, 10))
        milk_y = int(self.height * frame_num / self.num_frames)
        
        # 在画布上绘制牛奶流
        if self.mask[milk_y, milk_x]:
            # 添加牛奶流的扩散效果
            radius = 8  # 增大扩散半径
            milk_point = (milk_y, milk_x)
            self.milk_history.append(milk_point)
            
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    # 基于距离的衰减效果
                    distance = np.sqrt(dy**2 + dx**2)
                    intensity = max(0, 1 - distance/radius)
                    
                    ny, nx = milk_y + dy, milk_x + dx
                    if (0 <= ny < self.height and 
                        0 <= nx < self.width and 
                        self.mask[ny, nx]):
                        # 更复杂的填充算法
                        self.canvas[ny, nx] += 0.3 * intensity
    
    def _simulate_milk_spread(self):
        """
        模拟牛奶的物理扩散行为
        """
        # 限制历史记录长度
        if len(self.milk_history) > 20:
            self.milk_history = self.milk_history[-20:]
        
        # 基于历史点的扩散
        for point in self.milk_history:
            y, x = point
            spread_radius = 5
            for dy in range(-spread_radius, spread_radius+1):
                for dx in range(-spread_radius, spread_radius+1):
                    ny, nx = y + dy, x + dx
                    distance = np.sqrt(dy**2 + dx**2)
                    
                    if (0 <= ny < self.height and 
                        0 <= nx < self.width and 
                        self.mask[ny, nx] and 
                        distance < spread_radius):
                        # 衰减扩散效果
                        intensity = max(0, 1 - distance/spread_radius)
                        self.canvas[ny, nx] += 0.1 * intensity

def create_sample_edge_image():
    """
    创建一个示例边缘检测图像
    
    返回:
    numpy.ndarray: 生成的边缘检测图像
    """
    # 创建一个空白图像
    height, width = 600, 600
    edge_image = np.zeros((height, width), dtype=np.uint8)
    
    # 在图像中心绘制一个白色圆形
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 2 - 50
    
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
    
    # 创建圆形边缘
    edge_image[dist_from_center <= radius] = 255
    
    return edge_image

def visualize_latte_art(edge_image=None):
    """
    可视化拉花动画
    
    参数:
    edge_image (numpy.ndarray, optional): 边缘检测后的图像。如果为None，则生成示例图像
    """
    # 如果没有提供边缘图像，生成示例图像
    if edge_image is None:
        edge_image = create_sample_edge_image()
    
    # 创建拉花艺术对象
    latte_art = CoffeeLatteArt(edge_image)
    
    # 生成动画帧
    frames = latte_art.generate_latte_art()
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(frames[0], cmap='gray', animated=True)
    ax.set_title('latte-art')
    ax.axis('off')
    
    def update(frame):
        im.set_array(frame)
        return [im]
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=frames, 
        interval=50,  # 50毫秒间隔
        blit=True
    )
    
    plt.tight_layout()
    plt.show()

# 如果直接运行此脚本，生成并显示拉花艺术动画
if __name__ == "__main__":
    visualize_latte_art()