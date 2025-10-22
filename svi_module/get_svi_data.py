
import os
import time
import numpy as np
from shapely.geometry import Point
import io
import requests
from PIL import Image
from streetview import search_panoramas, get_panorama
import random
import geopandas


class StreetViewDownloader:
    def __init__(self, output_dir, batch_size=10, rate_limit_delay=1):
        """
        Initialize the downloader with output directory and batch size
        
        Args:
            output_dir (str): Directory to save images and metadata
            batch_size (int): Number of points to process in each batch
            rate_limit_delay (float): Delay between API requests in seconds
        """
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.image_dir = os.path.join(output_dir, 'images')
        self.metadata_file = os.path.join(output_dir, 'svi_info.csv')
        
        # Create necessary directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
        # Initialize metadata file with headers
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                f.write('lon,lat,pano_id,pano_lon,pano_lat,pano_heading,pano_pitch,pano_roll,status\n')

    def generate_points(self, bb, step=0.0005):
        """
        Generator function to yield grid points within the bounding box
        
        Args:
            bb (list): Bounding box coordinates [lat_min, lat_max, lon_min, lon_max]
            step (float): Grid step size in degrees
        """
        lat_range = np.arange(bb[0], bb[1], step)
        lon_range = np.arange(bb[2], bb[3], step)
        
        for lat in lat_range:
            for lon in lon_range:
                yield Point(lon, lat)

    def download_street_view_image(self, pano_id):
        """
        Download street view image using the pano id with rate limiting
        """
        file_path = os.path.join(self.image_dir, f'{pano_id}.jpg')

        if os.path.exists(file_path):
            return 'Duplicate'
        
        else:

            zoom = 3
            status = 'failed'
            try:
                # Get panorama image
                image_output =  get_panorama(pano_id, zoom)

                image_output.save(file_path, "jpeg")
                status = 'success'
                print(f"Downloaded image for pano_id: {pano_id}")
                time.sleep(1)
                
            except Exception as e:
                print(f"failed for pano_id {pano_id}: {str(e)}")
                
                #time.sleep(2 ** attempt)  # Exponential backoff
                    
            return status

    def process_point(self, point):
        """
        Process a single point and return its result
        
        Args:
            point (Point): Point object to process
            
        Returns:
            str: Result string for the point
        """
        lon, lat = point.x, point.y
        
        try:
            # Search for panoramas with retry logic
            panos = None
            for attempt in range(3):
                try:
                    time.sleep(self.rate_limit_delay)
                    panos = search_panoramas(lat, lon)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed to search panoramas: {str(e)}")
                    #time.sleep(2 ** attempt)  # Exponential backoff
            
            if not panos:
                return f'{lon},{lat},failed,NA,NA,NA,NA,NA,no_panorama_found'
            
            # Get first panorama
            pano = panos[-1]
            pano_id = pano.pano_id
            
            # Download image
            image_status = self.download_street_view_image(pano_id)
            
            return f'{lon},{lat},{pano_id},{pano.lon},{pano.lat},{pano.heading},{pano.pitch},{pano.roll},{image_status}'
            
        except Exception as e:
            return f'{lon},{lat},failed,NA,NA,NA,NA,NA,error:{str(e)}'

    def run(self, bb):
        """
        Main method to run the downloader sequentially
        
        Args:
            bb (list): Bounding box coordinates [lat_min, lat_max, lon_min, lon_max]
        """
        print("Starting download with sequential processing")
        
        points_generator = self.generate_points(bb)
        total_points_processed = 0
        batch_results = []
        
        while True:
            try:
                point = next(points_generator)
            except StopIteration:
                # Process any remaining results
                if batch_results:
                    with open(self.metadata_file, 'a') as f:
                        f.write('\n' + '\n'.join(batch_results) + '\n')
                print("All points processed")
                break
            
            result = self.process_point(point)
            batch_results.append(result)
            total_points_processed += 1
            
            # Write results to file when batch size is reached
            if len(batch_results) >= self.batch_size:
                with open(self.metadata_file, 'a') as f:
                    f.write('\n' + '\n'.join(batch_results) + '\n')
                print(f"Processed batch of {len(batch_results)} points. Total points processed: {total_points_processed}")
                batch_results = []

# Example usage
if __name__ == '__main__':
    # Define bounding box [lat_min, lat_max, lon_min, lon_max]
    
    # Replace 'path/to/your/shapefile.shp' with your shapefile's path
    gdf = geopandas.read_file('./raw_data/os_built_extent/glasgow_open_built_areas.shp')
    ##
    gdf = gdf.to_crs(4326)
    ##
    lat1, lon1, lat2, lon2 = gdf.total_bounds
    #BOUNDING_BOX = [55.71, 55.98, -4.54, -4.22]
    BOUNDING_BOX = [lon1, lon2, lat1, lat2]
    print(BOUNDING_BOX)

    # Initialize downloader with smaller batch size and rate limiting
    downloader = StreetViewDownloader(
        output_dir='./svi_module/svi_data/',
        batch_size=10,  # Smaller batch size for better progress tracking
        rate_limit_delay=0.7  # 1 second delay between API requests
    )
    
    # Run the downloader
    downloader.run(BOUNDING_BOX)