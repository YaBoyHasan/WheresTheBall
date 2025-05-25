import numpy as np
import matplotlib.pyplot as plt

data = np.load("C:/Users/haych/Desktop/1WheresTheBall/WheresTheBall/botb_data/processed_data.npz")
X, y = data['X'], data['y']

# Show 5 random samples
for i in range(5):
    idx = np.random.randint(0, len(X))
    img = (X[idx] * 255).astype(np.uint8)
    coords = y[idx]

    h, w = img.shape[:2]
    abs_x = int(coords[0] * w)
    abs_y = int(coords[1] * h)

    plt.imshow(img)
    plt.scatter([abs_x], [abs_y], c='red', s=40)
    plt.title(f"Sample {idx}")
    plt.show()
