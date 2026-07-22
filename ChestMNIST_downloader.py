from medmnist import ChestMNIST, VesselMNIST3D

# Download the 2D Chest X-Ray dataset
print("Downloading ChestMNIST...")
chest_train = ChestMNIST(split="train", download=True, size=28)


print(f"Datasets are downloaded and stored at: {chest_train.root}")