import qknn.kmeans.kmeans as kmeans

if __name__ == "__main__":
    s = kmeans.QKM(K=3, input_path="./kmeans_data.csv", feature_list=["Feature 1", "Feature 2"], output_path=None)
    s.train()