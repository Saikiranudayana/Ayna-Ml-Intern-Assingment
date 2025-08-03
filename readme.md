Project Title: Conditional Polygon Coloring using UNet (Image-to-Image Translation)
🔧 Internship Role: ML Engineer
This project is your assignment to demonstrate ML skills for an internship. You’re using a deep learning model to solve a conditional image generation problem:

“Given a grayscale image of a polygon and a color name (like red, green), generate the colored version of the image.”

This project teaches a model how to color grayscale polygon images based on a given color condition.

Think of this as:

You give the model a black-and-white polygon shape and a target color label (like "blue").

The model learns to paint that shape with the right color.

It does this by training on hundreds of examples where it sees grayscale input and the correct colored version.

🧱 Project Structure
1. Dataset
Stored in folders like training/inputs, training/outputs, and a data.json file.

Each entry has:

Input: grayscale polygon image

Output: same polygon, colored

Color condition: like "red", "green", etc.

9 total color categories are handled:
red, blue, green, yellow, orange, purple, cyan, magenta, black


<img width="1104" height="497" alt="Screenshot 2025-08-03 120816" src="https://github.com/user-attachments/assets/5e880d88-f3f4-4d3f-8a4f-775f6bc66036" />



2. Model Architecture – 🧠 UNet
The model is a UNet: a popular image-to-image neural network.

It takes:

A 64×64 grayscale polygon image

A one-hot encoded color condition

It outputs:

A 64×64 colored image


<img width="1088" height="733" alt="Screenshot 2025-08-03 120810" src="https://github.com/user-attachments/assets/e6b435a8-85ef-4e6f-bf57-26bc4e992534" />


3. Training Setup
Epochs: 100

Batch size: 16

Optimizer: Adam

Loss: Mean Squared Error (MSE)

Train-validation split: 80–20%

4. Color Conditioning
Each color is encoded into a numeric label (e.g., "red" = 0, "green" = 1).
The model learns which color to apply based on this condition.

📊 Visual Outputs
✅ 1. Confusion Matrix
🖼️ Insert Screenshot:
<img width="1127" height="709" alt="Screenshot 2025-08-03 120824" src="https://github.com/user-attachments/assets/bda68176-0cd4-40b7-8bf5-7a0367ad205e" />



“Diagonal boxes show correct predictions (e.g., red → red).”

📈 2. Accuracy and Loss Graph

<img width="1104" height="497" alt="Screenshot 2025-08-03 120816" src="https://github.com/user-attachments/assets/ee823a48-73b0-43b7-8715-e69846167cb0" />


“Training loss decreasing (good)”

🎨 3. Sample Predictions
Input: black polygon

Prediction: colored polygon

Ground truth: actual correct color

<img width="1089" height="573" alt="image" src="https://github.com/user-attachments/assets/ff742071-53c8-4d8c-85ac-a553672decb2" />


✅ Final Accuracy:
~90% validation accuracy in predicting correct colored outputs, based on the given condition and grayscale input.

🧾 Final Summary:
This assignment focuses on solving a conditional image generation task using deep learning. I built a UNet-based neural network that learns to color grayscale polygon shapes based on a specified color condition. The dataset consists of input-output image pairs and a color label. The model achieved ~90% validation accuracy, and the predictions are visually consistent with the expected outputs. This project demonstrates skills in PyTorch, image preprocessing, dataset creation, model training, and evaluation using metrics and visualizations.
