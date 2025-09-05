# cnn_phase_retrieval_full.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from skimage.filters import threshold_local
import matplotlib.image as mpimg

# Create clean project folder structure

BASE_DIR = os.path.join(os.getcwd(), "cnn_phase_retrieval")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
GIFS_DIR = os.path.join(BASE_DIR, "outputs", "gifs")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

for d in [PLOTS_DIR, GIFS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("All outputs will be saved under:", BASE_DIR)

# Generate synthetic holograms + objects

def generate_synthetic_data(num_samples=200, img_size=64):
    X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)  # hologram
    Y = np.zeros_like(X)  # object
    
    for i in range(num_samples):
        # Synthetic object: random circular pattern
        obj = np.zeros((img_size,img_size), dtype=np.float32)
        n_circles = np.random.randint(1,4)
        for _ in range(n_circles):
            radius = np.random.randint(3,10)
            cx = np.random.randint(radius,img_size-radius)
            cy = np.random.randint(radius,img_size-radius)
            y, x = np.ogrid[-cy:img_size-cy, -cx:img_size-cx]
            mask = x**2 + y**2 <= radius**2
            obj[mask] = np.random.uniform(0.5,1.0)
        obj /= np.max(obj) + 1e-8
        Y[i,:,:,0] = obj
        
        # Synthetic hologram: simple linear transform + noise
        holo = np.clip(obj + 0.1*np.random.randn(img_size,img_size), 0, 1)
        X[i,:,:,0] = holo
    return X, Y

print("Generating synthetic holograms and objects...")
X_train, Y_train = generate_synthetic_data(num_samples=200)
X_test, Y_test = generate_synthetic_data(num_samples=20)

# Visualize samples

def plot_samples(X, Y, num_samples=6, save_path=None):
    fig = plt.figure(figsize=(12,3))
    for i in range(num_samples):
        plt.subplot(2,num_samples,i+1)
        plt.imshow(X[i,:,:,0], cmap='gray')
        plt.axis('off')
        if i==0: plt.title("Input Hologram")
        plt.subplot(2,num_samples,num_samples+i+1)
        plt.imshow(Y[i,:,:,0], cmap='gray')
        plt.axis('off')
        if i==0: plt.title("Ground Truth Object")
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

plot_samples(X_train, Y_train, num_samples=6, save_path=os.path.join(PLOTS_DIR,"sample_inputs_labels.png"))

# Build CNN Encoder-Decoder

shap = (64,64,1)
inp = Input(shape=shap)
d1 = Conv2D(filters=20, kernel_size=3, strides=2, activation='relu', padding='same')(inp)
e1 = Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation='relu', padding='same')(d1)
model = Model(inp, e1)

# Save and display model plot

plot_model(model, to_file=os.path.join(PLOTS_DIR,'model.png'), show_shapes=True)

data = mpimg.imread(os.path.join(PLOTS_DIR,'model.png'))
plt.figure(figsize=(10,10))
plt.imshow(data)
plt.axis('off')
plt.title("CNN Model Architecture")
plt.savefig(os.path.join(PLOTS_DIR,"cnn_model_plot_display.png"), dpi=200)
plt.show()

# Compile and summarize

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()
with open(os.path.join(MODELS_DIR,"model_summary.txt"), "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Train CNN
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=30, batch_size=16, verbose=2)

# Save training loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CNN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR,"training_loss.png"), dpi=200)
plt.close()

# Evaluate and visualize predictions

y_pred = model.predict(X_test)

fig = plt.figure(figsize=(12,6))
for i in range(4):
    xte = X_test[i,:,:,0]
    yte = Y_test[i,:,:,0]
    pred = y_pred[i,:,:,0]
    
    # Local thresholding
    local_thresh = threshold_local(pred, block_size=11, offset=0.05)
    final_out = pred > local_thresh
    
    plt.subplot(4,4,i*4+1)
    plt.imshow(xte, cmap='gray'); plt.axis('off'); plt.title("Input")
    plt.subplot(4,4,i*4+2)
    plt.imshow(yte, cmap='gray'); plt.axis('off'); plt.title("Ground Truth")
    plt.subplot(4,4,i*4+3)
    plt.imshow(pred, cmap='gray'); plt.axis('off'); plt.title("CNN Output")
    plt.subplot(4,4,i*4+4)
    plt.imshow(final_out, cmap='gray'); plt.axis('off'); plt.title("Post-processed")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,"predictions.png"), dpi=200)
plt.show()

print("CNN phase retrieval project completed. All outputs saved under:", BASE_DIR)
