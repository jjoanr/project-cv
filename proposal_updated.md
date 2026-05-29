# Computer Vision: Short Project Definition

**Joan Rodríguez, Albert Casamitjana**

---

## 1. Project Description

This project aims to develop a computer vision system capable of recognizing and digitizing the state of a real chess board from a photograph, reconstructing the position of all pieces in digital format (equivalent to what platforms like chess.com display). Given a single image of a physical board, the system automatically detects the board, segments it into its 64 squares, classifies each square into one of 13 possible classes (empty or one of the six piece types in two colors), and outputs the resulting position in FEN notation along with a visual render. As a secondary objective, the system aims to generalize to non-standard chess piece designs through transfer learning.

## 2. Project Pipeline Design (with 50% milestone)

The pipeline is structured in four sequential phases: board detection and perspective correction (using OpenCV), segmentation into 64 individual squares, per-square classification via a CNN, and final reconstruction into FEN notation with a visual render. The first 50% milestone covers the full computer vision preprocessing pipeline plus the training and evaluation of the neural network for the standard piece recognition case, establishing a functional end-to-end system from image to FEN. The second 50% focuses on the digital board display and the extension of the model to handle alternative, non-standard chess piece designs through transfer learning.

## 3. Test Design and Image Database

For validation, the dataset is split into train (70%), validation (15%), and test (15%) subsets. The primary data source is the ChessRender360 dataset (Kaggle), which provides chess board images rendered from 360 degrees of viewpoints, covering diverse perspectives and lighting conditions across all piece types.

## 4. Generalization to Non-Standard Chess Piece Designs

Since non-standard sets may differ substantially in shape, texture, and color from standard Staunton pieces, the base CNN trained on ChessRender360 cannot be expected to generalize out-of-the-box. To address this, we explore the following strategies:

- **Transfer Learning:** The CNN backbone (e.g., ResNet-50 or EfficientNet) pre-trained on standard pieces is fine-tuned on a small labeled dataset of the target non-standard set. In a first stage, only the final classification layers are retrained (feature extraction mode), freezing the convolutional backbone to leverage already-learned low-level features (edges, textures, shapes). If sufficient data is available, a second stage unfreezes the full network for end-to-end fine-tuning with a low learning rate.

- **Few-Shot Learning:** When very few labeled examples of non-standard pieces are available, few-shot learning techniques can be considered. Approaches such as Siamese networks or prototypical networks learn a similarity metric rather than explicit class boundaries, enabling recognition from as few as 1–5 examples per class.

- **Data Augmentation:** To compensate for the small size of non-standard piece datasets, aggressive augmentation is applied during fine-tuning: random rotations, color jitter, perspective transforms, and synthetic shadow overlays. This increases effective dataset size and improves robustness to variations in lighting and viewpoint.

The expected outcome is a system that can adapt to a new piece style with as few as 10–50 labeled images per class, substantially reducing annotation effort compared to training from scratch.
