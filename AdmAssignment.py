import numpy as np
import scipy.sparse as sp
import itertools
import sys
import random
import math
import time
import hashlib  
from collections import defaultdict  

start_time = time.time()

#Function to read the data from the .npy file
def read_data_sparse_from_npy(file_path):
    data_array = np.load(file_path, allow_pickle= True)
    user_ids = np.array(data_array[:, 0].astype(int) - 1)
    movie_ids = np.array(data_array[:, 1].astype(int) - 1)
    ratings = np.array(data_array[:, 2])
    num_users = len(np.unique(user_ids))
    num_movies =  len(np.unique(movie_ids))
    return sp.csr_matrix((ratings, (user_ids, movie_ids)), shape=(num_users, num_movies))

#Generating random permutations
def generate_random_permutations(num_permutations, num_movies):
    return np.array([np.random.permutation(num_movies) for _ in range(num_permutations)])

#Function to compute the minhash signatures
def compute_minhash_signatures(sparse_matrix, permutations):
    
    sparse_matrix = sp.csr_matrix(sparse_matrix)
    num_permutations = permutations.shape[0]
    num_users,num_movies = sparse_matrix.shape
    signatures = np.full((num_permutations, num_users), np.inf)

    # print(f"Sparse matrix shape: {sparse_matrix.shape}")
    # print(f"Permutations shape: {permutations.shape}")
    # print(f"Number of movies: {sparse_matrix.shape[1]}")
    
    user_movies = sparse_matrix.indices  
    user_ptr = sparse_matrix.indptr      

    # Processing each user in the sparse matrix
    for user_idx in range(num_users):
        start_idx = user_ptr[user_idx]
        end_idx = user_ptr[user_idx + 1]
        movie_indices = user_movies[start_idx:end_idx]
        
        if len(movie_indices) == 0:
            continue
        
        # Computing the minhash signatures for the current user
        for perm_idx in range(num_permutations):
            permuted_indices = permutations[perm_idx, movie_indices]
            signatures[perm_idx, user_idx] = np.min(permuted_indices)
    
    return signatures


#Function to divide the MinHash signature matrix into bands using LSH
def lsh_band(signatures, num_bands):
    num_rows, num_users = signatures.shape
    rows_per_band = num_rows // num_bands
    buckets = defaultdict(list)

    for b in range(num_bands):
        start_row = b * rows_per_band
        end_row = start_row + rows_per_band
        band = signatures[start_row:end_row, :]
        for user_id, hash_value in enumerate(map(tuple, band.T)):
            buckets[hash_value].append(user_id)
    return buckets


#Function to compute the Jaccard similarity between two users
def compute_jaccard_similarity(user1, user2, matrix):
    set1 = set(matrix[user1].indices)
    set2 = set(matrix[user2].indices)
    return len(set1 & set2) / len(set1 | set2)

#Function to compute the similar pairs
def similarpairs(buckets, matrix, threshold):
    similar_pairs = set()
    for bucket_users in buckets.values():
        if len(bucket_users) > 1:
            for i in range(len(bucket_users)):
                for j in range(i + 1, len(bucket_users)):
                    u1, u2 = bucket_users[i], bucket_users[j]
                    if u1 < u2:
                        similarity = compute_jaccard_similarity(u1, u2, matrix)
                        if similarity > threshold: #Checking if their similarity is greater than the given threshold
                            similar_pairs.add((u1 + 1, u2 + 1))
    return similar_pairs


def main():
    if len(sys.argv) < 3:
        print("Usage: python AdmAssignment.py <data_file> <random_seed>")
        sys.exit(1)
    data_file = sys.argv[1]
    random_seed = int(sys.argv[2])
    random.seed(random_seed)
    np.random.seed(random_seed)

    num_permutations = 120
    num_bands = 15
    jaccard_threshold = 0.5
   

    print("Reading data...")
    sparse_matrix = read_data_sparse_from_npy(data_file)

    print("Generating random permutations...")
    num_movies = sparse_matrix.shape[1]
    permutations = generate_random_permutations(num_permutations, num_movies)
    print("Computing MinHash signatures...")
    signatures = compute_minhash_signatures(sparse_matrix, permutations)

    print("Applying LSH...")
    buckets = lsh_band(signatures, num_bands)

    print("Finding the similar pairs...")
    similar_pairs = similarpairs(buckets, sparse_matrix, jaccard_threshold)
    result_file = "similarpairs.txt"
    with open(result_file, "w") as f:
        for u1, u2 in sorted(similar_pairs):
            f.write(f"{u1},{u2}\n")
    print("The number of filtered pairs are: ",len(similar_pairs))
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"Execution completed. Results written to '{result_file}'.")
    print(f"Time taken: {elapsed_time_minutes:.2f} minutes")

if __name__ == "__main__":
    main()
