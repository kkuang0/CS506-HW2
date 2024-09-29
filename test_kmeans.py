# test_kmeans.py
import numpy as np
from kmeans import KMeans

def test_initialize_random():
    # Test the random initialization method
    kmeans = KMeans(k=3, init_method="random")
    kmeans.initialize()
    assert len(kmeans.centers) == 3, f"Expected 3 centers, got {len(kmeans.centers)}"
    print("Random Initialization Test Passed")

def test_initialize_farthest_first():
    # Test the farthest first initialization method
    kmeans = KMeans(k=3, init_method="farthest")
    kmeans.initialize()
    assert len(kmeans.centers) == 3, f"Expected 3 centers, got {len(kmeans.centers)}"
    print("Farthest First Initialization Test Passed")

def test_initialize_kmeans_plus_plus():
    # Test the kmeans++ initialization method
    kmeans = KMeans(k=3, init_method="kmeans++")
    kmeans.initialize()
    assert len(kmeans.centers) == 3, f"Expected 3 centers, got {len(kmeans.centers)}"
    print("KMeans++ Initialization Test Passed")

def test_clustering():
    # Test the clustering process
    kmeans = KMeans(k=3)
    kmeans.initialize()
    for _ in range(10):  # Perform 10 iterations
        kmeans.step()
        if kmeans.converged:
            break
    assert kmeans.converged, "KMeans did not converge in 10 steps"
    print("Clustering Test Passed")

def test_to_dict():
    # Test the to_dict() function
    kmeans = KMeans(k=3)
    kmeans.initialize()
    kmeans.step()
    result = kmeans.to_dict()
    assert 'data' in result and 'centers' in result and 'assignments' in result and 'converged' in result
    assert len(result['centers']) == 3, f"Expected 3 centers, got {len(result['centers'])}"
    print("to_dict() Test Passed")

if __name__ == "__main__":
    test_initialize_random()
    test_initialize_farthest_first()
    test_initialize_kmeans_plus_plus()
    test_clustering()
    test_to_dict()
    print("All tests passed!")
