import redis
import numpy as np
from redis.commands.search.query import Query

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def find_nearest_neighbors(redis_client, index_name, query_embedding, top_k=5):
    """
    Find top_k nearest neighbors for a given embedding in idx_mnist_vec index
    query_embedding: numpy array of shape (512,) with float32 values
    """
    # Convert numpy array to binary blob (exact format Redis expects)
    query_blob = query_embedding.astype(np.float32).tobytes()
    
    # Perform KNN vector search
    query_str = f"(-@label:[4 4])=>[KNN {top_k} @embedding $vec AS distance]"
    q = Query(query_str).return_fields("label", "distance").sort_by("distance").dialect(2)
    
    results = redis_client.ft(index_name).search(
        q,
        query_params={"vec": query_blob}
    )
    
    # Parse results
    neighbors = []
    for doc in results.docs:
        key = doc.id
        label = doc['label']
        distance = doc['distance']  # Lower = more similar (COSINE)
        neighbors.append({
            'key': key,
            'label': int(label),
            'distance': float(distance)
        })
    
    return neighbors

# Example usage
if __name__ == "__main__":
    # Example: Get embedding from existing document
    sample_key = "img:json:30"
    embedding_blob = r.json().get(sample_key, "$.embedding")
    if embedding_blob:
        # Convert list to numpy array
        query_embedding = np.array(embedding_blob[0], dtype=np.float32)
        
        # Find 5 nearest neighbors
        neighbors = find_nearest_neighbors(r, "idx_mnist_vec", query_embedding, top_k=5)
        
        print("Nearest neighbors:")
        for i, neighbor in enumerate(neighbors):
            print(f"{i+1}. Key: {neighbor['key']}, Label: {neighbor['label']}, Distance: {neighbor['distance']:.4f}")
    else:
        print("Sample key not found. Create some data first.")