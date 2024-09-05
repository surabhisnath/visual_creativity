import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np


# 1. Load the images from folders A and B
def load_images_from_folder(folders):
    images = []
    for folder in folders:
        for filename in [f for f in os.listdir(folder) if f[0] != "."][0:30]:
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert("L")  # 'L' mode is for grayscale
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue  # Skip the file if it cannot be opened
            images.append(img)
    return images


# Paths to folders
folders_A = [
    "../data/adults/stimuli_G",
    "../data/adults/stimuli_I",
    "../data/adults/stimuli_R",
]

folders_B = [
    "../data/children/stimuli_G",
    "../data/children/stimuli_I",
    "../data/children/stimuli_R",
]


images_A = load_images_from_folder(folders_A)
images_B = load_images_from_folder(folders_B)

# 2. Preprocess images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to the input size required by the CNN
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

preprocessed_A = [transform(img) for img in images_A]
preprocessed_B = [transform(img) for img in images_B]

# Stack the images into batches
batch_A = torch.stack(preprocessed_A)
batch_B = torch.stack(preprocessed_B)

# 3. Load a CNN model (using ResNet18 as an example)
model = models.resnet18(pretrained=True)
# Modify the first convolutional layer to accept 1-channel (grayscale) input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Remove the final fully connected layer
model = nn.Sequential(*list(model.children())[:-1])
model.eval()


# 4. Extract final layer representations
def extract_features(batch):
    with torch.no_grad():
        features = model(batch)
    return features.view(features.size(0), -1)  # Flatten the features


features_A = extract_features(batch_A)
features_B = extract_features(batch_B)

# 5. Dimensionality reduction using t-SNE
# Combine features for both A and B
features_all = torch.cat([features_A, features_B])

# Perform t-SNE to reduce the dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_features = tsne.fit_transform(features_all)

# Separate reduced features for A and B
reduced_A = reduced_features[: len(features_A)]
reduced_B = reduced_features[len(features_A) :]

# 6. Plot the representations with different colors for A and B
plt.figure(figsize=(8, 8))
plt.scatter(reduced_A[:, 0], reduced_A[:, 1], color="blue", label="adult drawings")
plt.scatter(reduced_B[:, 0], reduced_B[:, 1], color="red", label="children drawings")
plt.title("Final Layer Representations (t-SNE reduced to 2D)")
plt.legend()
plt.savefig("tsne.png")
