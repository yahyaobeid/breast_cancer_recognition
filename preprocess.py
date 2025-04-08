import os
import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_mias_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess MIAS PGM images and convert them to RGB format.
    
    Args:
        input_dir (str): Directory containing PGM images
        output_dir (str): Directory to save processed images
        target_size (tuple): Target size for resizing (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PGM files
    pgm_files = [f for f in os.listdir(input_dir) if f.endswith('.pgm')]
    logger.info(f"Found {len(pgm_files)} PGM files")
    
    for pgm_file in pgm_files:
        try:
            # Read PGM image
            img_path = os.path.join(input_dir, pgm_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logger.warning(f"Could not read image: {pgm_file}")
                continue
            
            # Resize image
            img_resized = cv2.resize(img, target_size)
            
            # Convert to RGB (3 channels)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values to [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # Save as PNG
            output_path = os.path.join(output_dir, pgm_file.replace('.pgm', '.png'))
            cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))
            
            logger.info(f"Processed: {pgm_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pgm_file}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_directory = "mias_images"  # Directory containing PGM files
    output_directory = "processed_images"  # Directory for processed images
    
    preprocess_mias_images(input_directory, output_directory) 