import random
import redis
import numpy as np
from idx_tools import display_image

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_image_data_from_redis(index):
    """Retrieve label, pixels, and embedding from Redis for a given index.
    
    Args:
        index: The index of the image record
        
    Returns:
        tuple: (label, image_array, embedding_array) or None if not found
    """
    key = f"img:json:{index}"
    data = r.json().get(key)
    
    if data is None:
        return None
    
    label = data['label']
    pixels = np.array(data['pixels'], dtype=np.uint8).reshape(28, 28)
    
    return label, pixels

def get_total_records():
    """Get approximate count of img:json:* records in Redis."""
    keys = r.keys("img:json:*")
    return len(keys)

# Get total number of records
total_records = get_total_records()
print(f"Found {total_records} img:json:<n> records in Redis.")

if total_records == 0:
    print("No data found in Redis. Please run store_image_data.py first.")
    exit()

# Interactive image display loop
while True:
    user_input = input("\nEnter image index (or 'end' to quit): ")
    
    if user_input.lower() == "end":
        print("Exiting...")
        break
    
    try:
        index = int(user_input)
        result = get_image_data_from_redis(index)
        
        if result:
            label, image = result
            print(f"Index: {index}. Label: {label}. Image:")
            display_image(image)
        else:
            print(f"No data found for index {index}.")
    except ValueError:
        print("Invalid input. Please enter a number or 'end'.")
