import numpy as np
import rasterio
import cv2
from tqdm import tqdm
from noise import pnoise2



class Agent:
    def __init__(self, start_positions, profile):
        self.positions = np.array([self.calculate_second_position(start_pos) for start_pos in start_positions])
        self.previous_positions = np.array(start_positions)
        self.profile = np.array(profile)
        self.velocities = self.positions - self.previous_positions
        self.alpha = np.array([0.55, 0.55])
        self.histories = [[] for _ in range(len(start_positions))]

    def calculate_second_position(self, first_pos):
        possible_directions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)])
        direction = possible_directions[np.random.choice(len(possible_directions))]
        new_position = np.array(first_pos) + direction
        return new_position

    def update_velocity_and_position(self, provisional_positions, landscape):
        self.velocities = provisional_positions - self.positions

        new_positions = (2 - self.alpha) * self.positions + (self.alpha - 1) * self.previous_positions + self.alpha * self.velocities

        self.previous_positions = self.positions.copy()

        new_positions = np.round(new_positions)
        self.positions = np.clip(new_positions, 0, landscape.size - 1)

        for i in range(len(self.positions)):
            self.histories[i].append(self.positions[i].copy())


    def reset(self):
            self.history = [self.positions]

    def random_walk(self, landscape):
        directions = np.random.randint(-1, 2, size=(len(self.positions), 2))
        new_positions = self.positions + directions
        return np.clip(new_positions, 0, landscape.size - 1)


    def route_travel(self, landscape):
        candidates = []

        for i, velocity in enumerate(self.velocities):
            velocity_directions = [(velocity[0], velocity[1])]

            # Append additional candidate directions based on the velocity direction
            if velocity_directions == [(1, 0)]:
                velocity_directions.extend([(1, 1), (1, -1)])
            elif velocity_directions == [(-1, 0)]:
                velocity_directions.extend([(-1, 1), (-1, -1)])
            elif velocity_directions == [(0, 1)]:
                velocity_directions.extend([(1, 1), (-1, 1)])
            elif velocity_directions == [(0, -1)]:
                velocity_directions.extend([(1, -1), (-1, -1)])
            elif velocity_directions == [(1, 1)]:
                velocity_directions.extend([(0, 1), (1, 0)])
            elif velocity_directions == [(-1, -1)]:
                velocity_directions.extend([(-1, 0), (0, -1)])
            elif velocity_directions == [(-1, 1)]:
                velocity_directions.extend([(-1, 0), (0, 1)])
            elif velocity_directions == [(1, -1)]:
                velocity_directions.extend([(1, 0), (0, -1)])

            # Check for candidates within the bounds of the landscape
            for dx, dy in velocity_directions:
                nx, ny = self.positions[i][0] + dx, self.positions[i][1] + dy
                nx = round(nx)
                ny = round(ny)
                if 0 <= nx < landscape.size and 0 <= ny < landscape.size:
                    if landscape.linear_features[nx, ny] == 1:
                        candidates.append((nx, ny))

            # Choose a position for the current agent
            if candidates:
                chosen_position = np.random.choice(len(candidates))
                self.positions[i] = np.array(candidates[chosen_position])
            else:
                # Fallback to random walk if no candidates
                self.positions[i] = self.random_walk(landscape)[i]

        return self.positions

    def direction_travel(self, landscape):
        new_positions = self.positions + self.velocities
        new_positions = np.clip(new_positions, 0, landscape.size - 1)
        self.positions = new_positions
        return self.positions

    def stay_put(self, landscape):
        return self.positions
    
    def view_enhance(self, landscape):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for i, (current_x, current_y) in enumerate(self.positions):
            current_x = int(current_x)
            current_y = int(current_y)
            current_elevation = landscape.elevation[current_x, current_y]
            best_position = self.positions[i]
            best_elevation = current_elevation

            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < landscape.size and 0 <= ny < landscape.size:
                    elevation = landscape.elevation[nx, ny]
                    if elevation > best_elevation:
                        best_position, best_elevation = [nx, ny], elevation

            # Update the agent's position
            self.positions[i] = np.array(best_position)
        
        return self.positions
    
    def backtrack(self, landscape):
        for i in range(len(self.positions)):
            if len(self.histories[i]) > 1:
                # Remove the last position and go back to the previous one
                self.histories[i].pop()
                self.positions[i] = np.array(self.histories[i][-1])
        return self.positions


class Landscape:
    def __init__(self, size, rgb_image_path=None, depth_image_path=None):
        self.size = size
        if rgb_image_path and depth_image_path:
            rgb_image = self.load_rgb_image(rgb_image_path)
            depth_image = self.load_depth_image(depth_image_path)
            self.rgb_image_cropped = self.crop_center(rgb_image, size, size)
            depth_image_cropped = self.crop_center(depth_image, size, size)
            self.elevation = self.process_depth(depth_image_cropped)
            self.linear_features = self.detect_linear_features(self.elevation)
            self.inaccessible_mask = self.detect_water_in_rgb(self.rgb_image_cropped)
        else:
            self.elevation = self.generate_elevation()
            self.linear_features = self.generate_linear_features()
            self.inaccessible_mask = self.generate_inaccessible_features()

    def load_rgb_image(self, rgb_image_path):
        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        if rgb_image is None or rgb_image.shape[2] != 3:
            raise ValueError("Invalid RGB image. Ensure the image is 3-channel (RGB).")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def load_depth_image(self, depth_image_path):
        with rasterio.open(depth_image_path) as dataset:
            depth_image = dataset.read(1).astype(np.float32)
        return depth_image

    def crop_center(self, img, cropx, cropy):
        y, x = img.shape[:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty+cropy, startx:startx+cropx]

    def process_depth(self, depth_image):
        depth_resized = cv2.resize(depth_image, (self.size, self.size))
        normalized_depth = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())
        return normalized_depth

    def detect_linear_features(self, elevation):
        edges = cv2.Canny((elevation * 255).astype(np.uint8), 100, 200)
        linear_features = cv2.resize(edges, (self.size, self.size)).astype(np.float32) / 255
        return linear_features

    def detect_water_in_rgb(self, rgb_image):
        # Assuming water is represented by blueish color, we use a range for blue.
        lower_blue = np.array([0, 0, 100])  # Lower bound for blue in RGB
        upper_blue = np.array([100, 150, 255])  # Upper bound for blue in RGB

        # Create a mask where the blueish areas are marked as 1, and others as 0
        mask = cv2.inRange(rgb_image, lower_blue, upper_blue)
        return mask

    def generate_elevation(self):
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        elevation = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                elevation[i][j] = pnoise2(i / scale,
                                          j / scale,
                                          octaves=octaves,
                                          persistence=persistence,
                                          lacunarity=lacunarity,
                                          repeatx=self.size,
                                          repeaty=self.size,
                                          base=0)
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        return elevation

    def generate_linear_features(self):
        linear_features = np.zeros((self.size, self.size))
        num_lines = np.random.randint(5, 15)
        for _ in range(num_lines):
            x_start = np.random.randint(0, self.size)
            y_start = np.random.randint(0, self.size)
            line_length = np.random.randint(10, 20)
            direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
            for i in range(line_length):
                if direction == 'horizontal':
                    x = x_start + i
                    y = y_start
                elif direction == 'vertical':
                    x = x_start
                    y = y_start + i
                elif direction == 'diagonal':
                    x = x_start + i
                    y = y_start + (i // 2)
                if 0 <= x < self.size and 0 <= y < self.size:
                    linear_features[x, y] = 1
        return linear_features



def simulate(landscape, agents, timesteps, visualize=False):
    agents.reset()

    choices = np.array(['RW', 'RT', 'DT', 'SP', 'VE', 'BT'])

    behavior_choices_matrix = np.random.choice(choices, p=agents.profile, size=timesteps*len(agents.positions)).reshape(timesteps, len(agents.positions))
        
    # behavior_choices_matrix.shape == (1000, 500)

    for i in tqdm(range(timesteps)):
        provisional_positions = np.zeros_like(agents.positions)
        # provisional_positions.shape == 500 * 2
        mask_rw = behavior_choices_matrix[i] == 'RW'
        provisional_positions[mask_rw] = agents.random_walk(landscape)[mask_rw]
        mask_rw = behavior_choices_matrix[i] == 'RT'
        provisional_positions[mask_rw] = agents.route_travel(landscape)[mask_rw]
        mask_rw = behavior_choices_matrix[i] == 'DT'
        provisional_positions[mask_rw] = agents.direction_travel(landscape)[mask_rw]
        mask_rw = behavior_choices_matrix[i] == 'SP'
        provisional_positions[mask_rw] = agents.stay_put(landscape)[mask_rw]
        mask_rw = behavior_choices_matrix[i] == 'VE'
        provisional_positions[mask_rw] = agents.view_enhance(landscape)[mask_rw]
        mask_rw = behavior_choices_matrix[i] == 'BT'
        provisional_positions[mask_rw] = agents.backtrack(landscape)[mask_rw]
        

        agents.update_velocity_and_position(provisional_positions, landscape)

