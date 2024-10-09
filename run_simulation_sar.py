import numpy as np
import os
import argparse
from itertools import product
from tqdm import tqdm
from behavior_profile_sar import Agent, Landscape, simulate

def generate_profiles():
    """Generate all possible probability profiles that add up to 1 with increments of 1/6."""
    increment = 1/6
    possibilities = [0, increment, 2*increment, 3*increment, 4*increment, 5*increment, 1]
    profiles = []
    
    # Generate all possible 6-value combinations where the sum is 1
    for p in product(possibilities, repeat=6):
        if np.isclose(sum(p), 1.0):
            profiles.append(p)
    return profiles

def calculate_energy_statistic(end_points, find_point):
    """
    Calculate the energy statistic for a given set of end points and the find point.
    
    end_points: List of end points where the agent finished.
    find_point: The "find" point for the specific agent's goal.
    """
    end_points = np.array(end_points, dtype=np.float32)
    find_point = np.array(find_point, dtype=np.float32)
    
    # Calculate the average distance between each end point and the find point.
    avg_dist_end_to_find = np.mean(np.linalg.norm(end_points - find_point, axis=1))
    
    # Calculate the average Euclidean distance between all pairs of end points.
    avg_dist_between_ends = np.mean([
        np.linalg.norm(ep1 - ep2) 
        for i, ep1 in enumerate(end_points) 
        for ep2 in end_points[i+1:]
    ])
    
    # print("End Points:", end_points)
    # print("Average Distance Between End Points:", avg_dist_between_ends)
    
    energy_stat = 2 * avg_dist_end_to_find - avg_dist_between_ends
    return energy_stat, avg_dist_end_to_find

def run_simulation_for_profile(profile, image_paths, start_points, find_points, timesteps, iterations, i):
    """Run the simulation for a given profile on all images and compute statistics."""
    weights = []
    for img_id, (naip_img_path, dem_img_path) in image_paths.items():
        start_pos = start_points[img_id]
        find_point = find_points[img_id]
        
        end_points = []
        
        for _ in tqdm(range(iterations)):
            landscape = Landscape(size=447, rgb_image_path=naip_img_path, depth_image_path=dem_img_path)
            agent = Agent(start_position=start_pos, profile=profile)
            simulate(landscape, agent, timesteps, visualize=False)
            end_points.append(agent.position)
        
        # Calculate energy statistic and the best profile
        energy_stat, avg_dist = calculate_energy_statistic(end_points, find_point)
        
        if energy_stat != 0: 
            weight = (avg_dist / energy_stat) ** 0.5
            weights.append(weight)
    
    return np.mean(weights) if weights else 0

def main(directory, start_points, find_points, timesteps=1000, iterations=500):
    profiles = generate_profiles()
    
    # Collect image paths (NAIP and DEM) from the directory
    image_paths = {}
    for file in os.listdir(directory):
        if file.startswith("NAIP_") and file.endswith(".tif"):
            img_id = file.split("_")[1].split(".")[0]  # Extract the number part of the file
            naip_img_path = os.path.join(directory, file)
            dem_img_path = os.path.join(directory, f"DEM_{img_id}.tif")
            if os.path.exists(dem_img_path):
                image_paths[img_id] = (naip_img_path, dem_img_path)
    
    # Track the best profile for each image
    profile_weights = np.zeros(len(profiles), dtype=np.float32)
    
    for i, profile in enumerate(tqdm(profiles)):
        weight = run_simulation_for_profile(profile, image_paths, start_points, find_points, timesteps, iterations, i)
        profile_weights[i] = weight

    # Normalize the profile weights to be between 0 and 1
    if profile_weights.sum() > 0:
        profile_weights /= profile_weights.sum()
    
    # Find the final profile with the normalized weights
    final_profile = np.dot(profile_weights, np.array(profiles))
    
    print("Final profile:", final_profile)
    return final_profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run agent simulation over NAIP and DEM images.')
    parser.add_argument('directory', type=str, help='Directory containing NAIP_####.tif and DEM_####.tif files')
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
