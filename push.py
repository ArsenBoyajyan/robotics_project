import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class ShapePushingSimulation:
    def __init__(self):
        # 随机种子
        np.random.seed(42)
        
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # 固定点（圆心）
        self.fixed_point = np.array([0, 0])
        
        # 有效区域半径（8cm）
        self.valid_radius = 8
        
        # 随机生成形状
        self.generate_shape()
        
        # 机械臂末端初始位置
        self.arm_endpoint = np.array([15.0, 15.0])  # Ensure dtype is float

        
        # 推动速度
        self.push_speed = 0.5
        
        # 绘图区域
        self.setup_plot()

    def generate_shape(self):
        # 随机选择形状
        if np.random.random() < 0.5:
            # 圆形（半径3-5cm）
            self.shape_type = 'circle'
            self.shape_size = np.random.uniform(3, 5)
            self.shape_center = np.array([np.random.uniform(-30, 30), 
                                           np.random.uniform(-30, 30)])
        else:
            # 正方形（边长5-8cm）
            self.shape_type = 'square'
            self.shape_size = np.random.uniform(5, 8)
            self.shape_center = np.array([np.random.uniform(-30, 30), 
                                           np.random.uniform(-30, 30)])

    def setup_plot(self):
        
        self.ax.clear()
        
        
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        
        
        self.ax.plot(self.fixed_point[0], self.fixed_point[1], 'ro', markersize=10, label='Fixed Point')
        
        # 绘制机械臂末端
        self.ax.plot(self.arm_endpoint[0], self.arm_endpoint[1], 'go', markersize=10, label='Arm Endpoint')
        
        # 绘制有效区域
        valid_region = Circle(self.fixed_point, self.valid_radius, 
                              facecolor='green', alpha=0.2, edgecolor='green')
        self.ax.add_patch(valid_region)
        
        # 绘制形状
        if self.shape_type == 'circle':
            shape = Circle(self.shape_center, radius=self.shape_size, 
                           facecolor='blue', edgecolor='blue')
        else:
            shape = Rectangle(self.shape_center - self.shape_size/2, 
                              width=self.shape_size, height=self.shape_size, 
                              facecolor='blue', edgecolor='blue')
        self.ax.add_patch(shape)
        
        
        self.ax.set_title('Placment Process')
        self.ax.set_xlabel('X-axis (cm)')
        self.ax.set_ylabel('Y-axis (cm)')
        self.ax.legend()
        
        
        self.ax.set_aspect('equal')

    def push_shape(self):
        # Calculate the vector from the shape center to the fixed point
        to_fixed = self.fixed_point - self.shape_center
        
        # Normalize the direction vector
        move_distance = np.linalg.norm(to_fixed)
        if move_distance == 0:  # Avoid division by zero
            return
        direction = to_fixed / move_distance

        # Calculate the desired position for the arm endpoint (opposite side of the coffee cup)
        arm_target_position = self.shape_center - direction * (self.shape_size / 2 + 2.5)  # Adjust distance as needed

        # Move the arm endpoint toward the target position
        arm_movement = arm_target_position - self.arm_endpoint
        arm_distance = np.linalg.norm(arm_movement)
        if arm_distance < self.push_speed:
            self.arm_endpoint = arm_target_position
        else:
            self.arm_endpoint += arm_movement / arm_distance * self.push_speed

        # Only push the shape if the arm endpoint is correctly positioned
        if np.linalg.norm(self.arm_endpoint - arm_target_position) < 0.1:
            if move_distance < self.push_speed:
                self.shape_center = self.fixed_point.copy()
            else:
                # Move the shape toward the fixed point
                self.shape_center += direction * self.push_speed

    def check_placement_validity(self):
        # 计算形状中心到固定点的距离
        dist = np.linalg.norm(self.shape_center - self.fixed_point)
        
        # 检查距离是否在有效范围内（不超过0.5cm）
        return dist <= (self.valid_radius)

    def run_simulation(self):
        # 最大迭代次数，防止无限循环
        max_iterations = 200
        iterations = 0
    
        # 存储每一步的状态
        positions = []
    
        while not self.check_placement_validity() and iterations < max_iterations:
            # 记录当前位置
            positions.append({
                'shape_center': self.shape_center.copy(),
                'arm_endpoint': self.arm_endpoint.copy()
            })
            
            
            self.push_shape()
            
            
            self.visualize_path(positions)
            
            
            plt.pause(0.01)  # 暂停0.1秒
            
            iterations += 1

    def visualize_path(self, positions):
        
        self.ax.clear()
        
        
        self.setup_plot()
        
        # 绘制路径
        shape_path = np.array([pos['shape_center'] for pos in positions])
        arm_path = np.array([pos['arm_endpoint'] for pos in positions])
        
        # 检查 shape_path 和 arm_path 是否是二维数组
        if shape_path.ndim == 1:
            shape_path = shape_path.reshape(-1, 2)
        if arm_path.ndim == 1:
            arm_path = arm_path.reshape(-1, 2)

        # 绘制路径轨迹
        self.ax.plot(arm_path[:, 0], arm_path[:, 1], 'g--', label='Arm Endpoint Path')
        
        # 最终位置
        self.ax.plot(self.shape_center[0], self.shape_center[1], 'bo', label='Final Shape Position')
        
        
        self.ax.legend()
        
        
        plt.draw()  


if __name__ == "__main__":
    sim = ShapePushingSimulation()
    sim.run_simulation()
    plt.show()  
