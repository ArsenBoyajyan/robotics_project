import cv2
import numpy as np
import pygame
from scipy.ndimage import gaussian_filter

class CoffeeLatteArt:
    def __init__(self, edge_image, num_frames=120):
        """
        Initialize the latte art animation generator using SPH simulation.

        Parameters:
        edge_image (numpy.ndarray): Edge-detected circular image.
        num_frames (int): Number of animation frames.
        """
        # Detect circular region
        self.mask = self._find_circular_region(edge_image)

        # Initialize canvas
        self.canvas = np.zeros_like(edge_image, dtype=np.float32)
        self.canvas[self.mask] = 0.1  # Initial background intensity

        # Animation parameters
        self.num_frames = num_frames
        self.width = edge_image.shape[1]
        self.height = edge_image.shape[0]

        # SPH particle system parameters
        self.num_particles = 2000
        self.particles = self._initialize_particles()

        # Milk diffusion parameters
        self.milk_history = []

    def _initialize_particles(self):
        particles = np.zeros((self.num_particles, 6))  # [y, x, vel_y, vel_x, density, intensity]
        for i in range(self.num_particles):
            attempts = 0
            while attempts < 100:
                y = np.random.randint(0, self.height)
                x = np.random.randint(0, self.width)
                if self.mask[y, x]:
                    particles[i, 0] = y
                    particles[i, 1] = x
                    break
                attempts += 1
            if attempts == 100:
                particles[i, 0] = self.height // 2
                particles[i, 1] = self.width // 2
        
        # Debug: Print particle positions
        for i, particle in enumerate(particles):
            print(f"Initialized Particle {i}: Position=({particle[0]}, {particle[1]})")
        return particles


    def _find_circular_region(self, edge_image):
        """
        Identify the circular region in the image.

        Parameters:
        edge_image (numpy.ndarray): Edge-detected image.

        Returns:
        numpy.ndarray: Boolean circular region mask.
        """
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
        mask = np.zeros_like(edge_image, dtype=bool)
        if circles is not None:
            circle = circles[0, 0]
            center_x, center_y, radius = map(int, circle)
            Y, X = np.ogrid[:edge_image.shape[0], :edge_image.shape[1]]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = dist_from_center <= radius
        else:
            height, width = edge_image.shape
            Y, X = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 2 - 10
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = dist_from_center <= radius

        # Debug: Print mask information here
        print(f"Mask created with {np.sum(mask)} pixels set to True (circular region).")
        return mask


    def _smoothing_kernel(self, distance, h=20):
        """
        Wendland kernel function.

        Parameters:
        distance (float): Distance between particles.
        h (float): Smoothing length.

        Returns:
        float: Kernel value.
        """
        q = distance / h
        if q <= 1.0:
            return (315 / (64 * np.pi * h**3)) * (1 - q)**3
        return 0

    def _compute_particle_density(self):
        """
        Compute density for all particles.
        """
        for i in range(self.num_particles):
            density = 0
            for j in range(self.num_particles):
                dy = self.particles[i, 0] - self.particles[j, 0]
                dx = self.particles[i, 1] - self.particles[j, 1]
                distance = np.sqrt(dy**2 + dx**2)
                density += self._smoothing_kernel(distance)
            self.particles[i, 4] = density

    def _apply_sph_dynamics(self, frame_num):
        self._compute_particle_density()

        center_x = self.width // 2
        amplitude = self.width // 8
        frequency = 0.2
        milk_x = int(center_x + amplitude * np.sin(frequency * frame_num) + np.random.randint(-10, 10))
        milk_y = int(self.height * frame_num / self.num_frames)

        for i in range(self.num_particles):
            dy = self.particles[i, 0] - milk_y
            dx = self.particles[i, 1] - milk_x
            distance = np.sqrt(dy**2 + dx**2)
            if distance < 20:
                self.particles[i, 2] += 0.1 * (milk_y - self.particles[i, 0]) / (distance + 1)
                self.particles[i, 3] += 0.1 * (milk_x - self.particles[i, 1]) / (distance + 1)
                intensity = max(0, 1 - distance / 20)
                self.particles[i, 5] = intensity

        # Debug: Print particle states after applying SPH dynamics
        for i, particle in enumerate(self.particles):
            print(f"Frame {frame_num}, Particle {i}: Position=({particle[0]:.2f}, {particle[1]:.2f}), "
                f"Velocity=({particle[2]:.2f}, {particle[3]:.2f}), Density={particle[4]:.2f}, "
                f"Intensity={particle[5]:.2f}")


    def generate_frame(self, frame_num):
        self._apply_sph_dynamics(frame_num)
        canvas_copy = self.canvas.copy()
        for particle in self.particles:
            y, x = int(particle[0]), int(particle[1])
            if 0 <= y < self.height and 0 <= x < self.width and self.mask[y, x]:
                canvas_copy[y, x] += 0.1 * particle[5]
        frame = gaussian_filter(canvas_copy, sigma=1.5)

        # Debug: Log frame statistics
        print(f"Frame {frame_num}: Min={frame.min():.4f}, Max={frame.max():.4f}")

        return (frame - frame.min()) / (frame.max() - frame.min())



def visualize_latte_art_pygame(edge_image):
    """
    Visualize latte art animation using Pygame.

    Parameters:
    edge_image (numpy.ndarray): Edge-detected image.
    """
    pygame.init()
    height, width = edge_image.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Latte Art Animation")

    latte_art = CoffeeLatteArt(edge_image)
    clock = pygame.time.Clock()
    running = True
    frame_num = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        

        frame = latte_art.generate_frame(frame_num)
        frame_surface = pygame.surfarray.make_surface((frame * 255).astype(np.uint8))
        screen.blit(pygame.transform.scale(frame_surface, (width, height)), (0, 0))
        pygame.display.flip()

        frame_num = (frame_num + 1) % latte_art.num_frames
        clock.tick(30)

    pygame.quit()


def create_sample_edge_image():
    """
    Create a sample edge-detected image.

    Returns:
    numpy.ndarray: Generated edge-detected image.
    """
    height, width = 600, 600
    edge_image = np.zeros((height, width), dtype=np.uint8)
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 2 - 50
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_image[dist_from_center <= radius] = 255
    return edge_image


if __name__ == "__main__":
    # Use the uploaded Sobel image or create a sample image
    edge_image = cv2.imread("sobel_combined_result.png", cv2.IMREAD_GRAYSCALE)
    if edge_image is None:
        print("No image found, generating a sample image.")
        edge_image = create_sample_edge_image()

    visualize_latte_art_pygame(edge_image)
