import numpy as np
from tqdm import tqdm
import os
import cv2
import rasterio
from itertools import product
import argparse
from vectorized_behavior_profile import Agent, Landscape, simulate
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import Manager
import time

def generate_profiles():
    """Generate all possible probability profiles that add up to 1 with increments of 1/6."""
    increment = 1/6
    possibilities = [0, increment, 2*increment, 3*increment, 4*increment, 5*increment, 1]
    profiles = np.array([p for p in product(possibilities, repeat=6) if np.isclose(sum(p), 1.0)], dtype=np.float32)
    return profiles


def calculate_energy_statistic(end_points, find_point):
    """
    Calculate the energy statistic for a batch of agents.
    """
    end_points = np.array(end_points, dtype=np.float32)
    find_point = np.array(find_point, dtype=np.float32)

    avg_dist_end_to_find = np.mean(np.linalg.norm(end_points - find_point, axis=1))
    avg_dist_between_ends = np.mean(np.linalg.norm(end_points[:, None] - end_points[None, :], axis=2))

    energy_stat = 2 * avg_dist_end_to_find - avg_dist_between_ends
    return energy_stat, avg_dist_end_to_find


def process_image(naip_img_path, dem_img_path, profile, start_pos, find_point, timesteps, iterations, i):
    """Process a single image, running the simulation and calculating the weight."""
    start_positions = np.tile(start_pos, (iterations, 1))

    agents = Agent(start_positions, profile)
    landscape = Landscape(size=447, rgb_image_path=naip_img_path, depth_image_path=dem_img_path)
    
    start_time = time.time()
    # Run the simulation for all agents in parallel
    simulate(landscape, agents, timesteps)
    print(str(i) + ":", time.time() - start_time, "seconds")
    # Calculate energy statistic for all iterations in batch
    energy_stat, avg_dist = calculate_energy_statistic(agents.positions, find_point)

    if energy_stat != 0:
        weight = (avg_dist / energy_stat) ** 0.5
        return weight
    return 0


def run_simulation_for_profile(profile, image_paths, start_points, find_points, timesteps, iterations, i):
    """Run the simulation for a given profile on all images and compute statistics."""
    weights = []

    def process_image_thread(img_id):
        naip_img_path, dem_img_path = image_paths[img_id]
        start_pos = np.array(start_points[img_id], dtype=np.float32)
        find_point = np.array(find_points[img_id], dtype=np.float32)
        return process_image(naip_img_path, dem_img_path, profile, start_pos, find_point, timesteps, iterations, i)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image_thread, image_paths.keys()))
        weights.extend(results)
        

    return np.mean(weights) if weights else 0


def run_simulation_for_profile_multiprocess(args):
    """Wrapper function for multiprocessing."""
    return run_simulation_for_profile(*args)


def main(directory, start_points, find_points, timesteps=1000, iterations=500):
    profiles = generate_profiles()

    image_paths = {}
    for file in os.listdir(directory):
        if file.startswith("NAIP_") and file.endswith(".tif"):
            img_id = file.split("_")[1].split(".")[0]
            naip_img_path = os.path.join(directory, file)
            dem_img_path = os.path.join(directory, f"DEM_{img_id}.tif")
            if os.path.exists(dem_img_path):
                image_paths[img_id] = (naip_img_path, dem_img_path)

    profile_weights = np.zeros(len(profiles), dtype=np.float32)

    args_list = [(profile, image_paths, start_points, find_points, timesteps, iterations, i) for i, profile in enumerate(profiles)]

    print(mp.cpu_count())
    with mp.Pool(processes=8) as pool:
        results = list(pool.imap(run_simulation_for_profile_multiprocess, args_list))

    profile_weights = np.array(results, dtype=np.float32)

    if profile_weights.sum() > 0:
        profile_weights /= profile_weights.sum()

    final_profile = np.dot(profile_weights, np.array(profiles))

    print("Final profile:", final_profile)
    return final_profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run agent simulation over NAIP and DEM images.')
    parser.add_argument('--directory', type=str, default='tif_images', help='Directory containing NAIP_####.tif and DEM_####.tif files')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps for each iteration')
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations per image')

    args = parser.parse_args()

    # IPPs, have to replace with actual data
    start_points = {
        '1': [230., 125.],
    }

    # Found positions, also have to replace with actual data
    find_points = {
        '1': [250., 130.],
    }

    main(args.directory, start_points, find_points, args.timesteps, args.iterations)
