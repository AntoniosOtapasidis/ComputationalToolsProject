import kagglehub

# Download latest version
path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset")

print("Path to dataset files:", path)
