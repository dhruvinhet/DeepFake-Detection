
## 📊 Visual Overview

High-level overview of the system's interaction from input to backend:

![Visual Overview](https://github.com/user-attachments/assets/c1add67a-bf0e-4260-b0b4-5318e3959d4e)

---

## 🔍 What Happens Under the Hood?

### 🖼️ Preprocessing Step

When the backend receives an image and triggers analysis, it first calls the `preprocess_image` function.

![Preprocessing Flow](https://github.com/user-attachments/assets/31912823-43b4-4b2e-a92c-c0340c3e3833)

---

### 🤖 Prediction Step

Once the image is preprocessed and passed to `predict_with_explanation`, this flow is executed:

![Prediction Flow](https://github.com/user-attachments/assets/da394a4e-2e82-4a6c-9779-7d4fa8047139)

---

### 📤 Response Generation

After prediction, the backend sends a JSON response with the result and generated image paths:

![Response Flow](https://github.com/user-attachments/assets/6ccc84ba-1f14-4c07-ae9a-5828a26d3bed)

---

## 🧠 What Happens During Training?

### 🔁 Training Cycle Explanation

During the `model.fit(...)` call, the following happens for every batch:

1. **Get Data:** Fetch a batch of images with real/fake labels.
2. **Model Predicts:** Use the current model to predict on the batch.
3. **Calculate Loss:** Compare predicted vs true using `binary_crossentropy`.
4. **Backpropagation:** Compute gradients of model parameters.
5. **Optimizer Step:** Adjust parameters using Adam optimizer.
6. **Repeat:** Continue for all batches in an epoch.
7. **Validation:** Evaluate on unseen validation data.
8. **Callbacks:** Monitor validation performance and apply early stopping if necessary.

### 🔄 Training Flow Diagram

![Training Flow](https://github.com/user-attachments/assets/42ff45b2-086d-4980-a13e-27f415bce03c)
