from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QRadioButton, QSpinBox,
                             QGraphicsView, QGraphicsScene, QFrame, QFileDialog, QGraphicsPixmapItem)
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
from glob import glob

class FaceRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 700)
        
        # Initialize face recognition system variables
        self.scaler = None
        self.pca = None
        self.clf = None
        self.X_test_p = None
        self.y_test = None
        self.paths_test = None
        self.current_image = None
        self.current_image_path = None
        
        # Main Layout
        main_layout = QHBoxLayout(self)
        
        # Left Panel (0.25 of screen width)
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_panel.setSpacing(10)
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setStretch(0, 1)
        
        # Buttons
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.detect_faces_btn = QPushButton("Detect Faces")
        self.detect_faces_btn.clicked.connect(self.detect_faces)
        # self.apply_pca_btn = QPushButton("Apply PCA")
        # self.apply_pca_btn.clicked.connect(self.apply_pca)
        self.recognize_faces_btn = QPushButton("Recognize Faces")
        self.recognize_faces_btn.clicked.connect(self.recognize_faces)
        
        # Face Recognition Result Display
        self.recognized_label = QLabel("Recognized Faces: ")
        self.recognized_text = QLabel("N/A")
        
        # Radio Buttons for Color/Grayscale
        self.color_radio = QRadioButton("Color")
        self.grayscale_radio = QRadioButton("Grayscale")
        self.grayscale_radio.setChecked(True)
        
        # # PCA Components Selector
        # self.pca_spinbox = QSpinBox()
        # self.pca_spinbox.setRange(1, 100)
        # self.pca_spinbox.setValue(10)
        
        # Performance Metrics
        self.performance_label = QLabel("Performance Accuracy:")
        self.performance_value = QLabel("N/A")
        self.precision_label = QLabel("Precision:")
        self.precision_value = QLabel("N/A")
        self.recall_label = QLabel("Recall:")
        self.recall_value = QLabel("N/A")
        
        # ROC Curve
        self.roc_label = QLabel("ROC")
        self.roc_display = QGraphicsView()
        self.roc_scene = QGraphicsScene()
        self.roc_display.setScene(self.roc_scene)
        self.roc_display.setFixedHeight(300)
        
        # Adding Widgets to Left Panel
        left_panel.addWidget(self.load_image_btn)
        left_panel.addWidget(self.detect_faces_btn)
        #left_panel.addWidget(self.apply_pca_btn)
        left_panel.addWidget(self.recognize_faces_btn)
        left_panel.addWidget(self.recognized_label)
        left_panel.addWidget(self.recognized_text)
        left_panel.addWidget(self.color_radio)
        left_panel.addWidget(self.grayscale_radio)
        # left_panel.addWidget(QLabel("Number of PCA Components:"))
        # left_panel.addWidget(self.pca_spinbox)
        left_panel.addWidget(self.performance_label)
        left_panel.addWidget(self.performance_value)
        left_panel.addWidget(self.precision_label)
        left_panel.addWidget(self.precision_value)
        left_panel.addWidget(self.recall_label)
        left_panel.addWidget(self.recall_value)
        left_panel.addWidget(self.roc_label)
        left_panel.addWidget(self.roc_display)
        
        # Right Panel (0.75 of screen width)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        # Input Image Viewport
        self.input_label = QLabel("Input")
        self.input_view = QGraphicsView()
        self.input_scene = QGraphicsScene()
        self.input_view.setScene(self.input_scene)
        self.input_view.setFrameShape(QFrame.Shape.Box)
        
        # Result / Eigenfaces Viewport
        self.result_label = QLabel("Result / Eigenfaces")
        self.result_view = QGraphicsView()
        self.result_scene = QGraphicsScene()
        self.result_view.setScene(self.result_scene)
        self.result_view.setFrameShape(QFrame.Shape.Box)
        
        right_panel.addWidget(self.input_label)
        right_panel.addWidget(self.input_view)
        right_panel.addWidget(self.result_label)
        right_panel.addWidget(self.result_view)
        
        # Adding Panels to Main Layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        
        self.setLayout(main_layout)
        
        # Initialize the face recognition system
        self.initialize_face_recognition()

    def initialize_face_recognition(self):
        """Initialize the face recognition system with the ORL dataset"""
        try:
            # Path to ORL data - you should update this path
            dataset_path = "C:\\Users\\Admin\\Documents\\CV Task 5\\CV Task 5\\archive"  
            
            # Load dataset
            X, y, paths = self.load_orl_dataset(dataset_path)
            print(f"Loaded {len(X)} images from {len(np.unique(y))} subjects.")
            
            # Split into train/test (80/20)
            X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
                X, y, paths, test_size=0.2, stratify=y, random_state=42)
            print(f"Train: {len(X_train)} images, Test: {len(X_test)} images")
            
            # Store test set for later evaluation
            self.y_test = y_test
            self.paths_test = paths_test
            
            # Preprocess
            X_train_s, self.scaler = self.preprocess(X_train)
            X_test_s, _ = self.preprocess(X_test, self.scaler)
            
            # PCA with initial components from spinbox
            #n_components = self.pca_spinbox.value()
            n_components = 10
            self.pca = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=42)
            X_train_p = self.pca.fit_transform(X_train_s)
            self.X_test_p = self.pca.transform(X_test_s)
            
            print(f"PCA: {self.pca.n_components_} components ({np.sum(self.pca.explained_variance_ratio_):.2%} variance)")
            
            # Train classifier
            self.clf = self.train_svm(X_train_p, y_train)
            
            # Initial evaluation
            self.evaluate_performance()
            
        except Exception as e:
            print(f"Error initializing face recognition system: {str(e)}")

    def load_orl_dataset(self, root_dir):
        """Load ORL dataset from directory"""
        X, y, paths = [], [], []
        for subj_dir in sorted(glob(os.path.join(root_dir, 's*'))):
            if not os.path.isdir(subj_dir):
                continue
            label = int(os.path.basename(subj_dir).lstrip('s'))
            for img_path in sorted(glob(os.path.join(subj_dir, '*.pgm'))):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                X.append(img)
                y.append(label)
                paths.append(img_path)
        if not X:
            raise RuntimeError(f"No images found in {root_dir}")
        return np.array(X), np.array(y), paths

    def preprocess(self, X, scaler=None):
        """Preprocess images with CLAHE and scaling"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        X_flat = np.array([clahe.apply(img).flatten() for img in X], dtype=np.float32)
        if scaler is None:
            scaler = StandardScaler().fit(X_flat)
        X_scaled = scaler.transform(X_flat)
        return X_scaled, scaler

    def train_svm(self, X, y):
        """Train SVM classifier"""
        clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        clf.fit(X, y)
        return clf


    def evaluate_performance(self):
        if self.clf is None or self.X_test_p is None or self.y_test is None:
            return

        y_pred = self.clf.predict(self.X_test_p)
        y_scores = self.clf.decision_function(self.X_test_p)
        
        # Convert y_true and y_pred to lists for easier manipulation
        y_true = list(self.y_test)
        y_pred = list(y_pred)
        labels = sorted(set(y_true))
        n_classes = len(labels)
        
        # One-hot encode y_true
        y_true_bin = [[1 if label == cls else 0 for cls in labels] for label in y_true]

        # Accuracy
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        accuracy = correct / len(y_true)

        # Precision and Recall (weighted)
        precision_total = 0
        recall_total = 0
        for cls in labels:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] == cls)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] != cls and y_pred[i] == cls)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] != cls)
            support = sum(1 for i in range(len(y_true)) if y_true[i] == cls)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            weight = support / len(y_true)
            
            precision_total += prec * weight
            recall_total += rec * weight

        # ROC Curve and AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, cls in enumerate(labels):
            # Sort scores and corresponding true labels
            scores = [row[i] for row in y_scores]
            true_bin = [row[i] for row in y_true_bin]
            combined = sorted(zip(scores, true_bin), reverse=True)

            tps = 0
            fps = 0
            P = sum(true_bin)
            N = len(true_bin) - P

            fpr_vals = []
            tpr_vals = []

            for score, true_val in combined:
                if true_val == 1:
                    tps += 1
                else:
                    fps += 1
                fpr_vals.append(fps / N if N else 0)
                tpr_vals.append(tps / P if P else 0)

            # Compute AUC using trapezoidal rule
            auc_val = 0.0
            for j in range(1, len(fpr_vals)):
                auc_val += (fpr_vals[j] - fpr_vals[j - 1]) * (tpr_vals[j] + tpr_vals[j - 1]) / 2

            fpr[i] = fpr_vals
            tpr[i] = tpr_vals
            roc_auc[i] = auc_val

        # Update UI values
        self.performance_value.setText(f"{accuracy:.2f}")
        self.precision_value.setText(f"{precision_total:.2f}")
        self.recall_value.setText(f"{recall_total:.2f}")

        # Plot ROC
        plt.figure()
        for i, cls in enumerate(labels):
            plt.plot(fpr[i], tpr[i], label=f'Class {cls} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Each Class')
        plt.legend(loc="lower right")

        # Save and display
        roc_img_path = "roc_curve_temp.png"
        plt.savefig(roc_img_path)
        plt.close()
        
        pixmap = QPixmap(roc_img_path)
        self.roc_scene.clear()
        self.roc_scene.addItem(QGraphicsPixmapItem(pixmap))



    def load_image(self):
        """Load an image for recognition"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.pgm *.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        if self.grayscale_radio.isChecked():
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(file_path)
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if img is None:
            print(f"Failed to load image: {file_path}")
            return
            
        self.current_image = img
        self.current_image_path = file_path
        
        # Convert to QImage and display
        height, width = display_img.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(display_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        
        self.input_scene.clear()
        self.input_scene.addItem(QGraphicsPixmapItem(q_pixmap))

    def detect_faces(self):
        """Detect faces in the current image"""
        if self.current_image is None:
            print("No image loaded")
            return
            
        # Convert to grayscale if needed
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        # Use OpenCV's Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around faces
        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if len(gray.shape) == 2 else self.current_image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display result
        height, width = result_img.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        
        self.result_scene.clear()
        self.result_scene.addItem(QGraphicsPixmapItem(q_pixmap))
        
        # Update recognized faces label
        self.recognized_text.setText(f"{len(faces)} faces detected")

    # def apply_pca(self):
    #     """Reapply PCA with new component count"""
    #     if self.scaler is None or self.clf is None:
    #         print("Face recognition system not initialized")
    #         return
            
    #     n_components = self.pca_spinbox.value()
    #     print(f"Reapplying PCA with {n_components} components")
        
    #     # Reload and preprocess the training data
    #     dataset_path = "C:\\Users\\Admin\\Documents\\CV Task 5\\CV Task 5\\archive"  # Replace with actual path to your dataset
    #     X, y, paths = self.load_orl_dataset(dataset_path)
    #     X_train, _, y_train, _, _, _ = train_test_split(
    #         X, y, paths, test_size=0.2, stratify=y, random_state=42)
        
    #     X_train_s, _ = self.preprocess(X_train, self.scaler)
        
    #     # Apply PCA with new component count
    #     self.pca = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=42)
    #     X_train_p = self.pca.fit_transform(X_train_s)
        
    #     # Retrain classifier
    #     self.clf = self.train_svm(X_train_p, y_train)
        
    #     # Re-evaluate performance
    #     self.evaluate_performance()

    def recognize_faces(self):
        """Recognize faces in the current image"""
        if self.current_image is None:
            print("No image loaded")
            return
            
        if self.scaler is None or self.pca is None or self.clf is None:
            print("Face recognition system not initialized")
            return
            
        # Convert to grayscale if needed
        if len(self.current_image.shape) == 3 and self.grayscale_radio.isChecked():
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        # Detect faces first
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            self.recognized_text.setText("No faces detected")
            return
            
        # Prepare result image
        result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if len(gray.shape) == 2 else self.current_image.copy()
        recognized_names = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to ORL dataset size (92x112)
            face_resized = cv2.resize(face_roi, (92, 112), interpolation=cv2.INTER_AREA)
            
            # Preprocess
            X_flat = np.array([face_resized.flatten()], dtype=np.float32)
            X_scaled = self.scaler.transform(X_flat)
            
            # Apply PCA
            X_pca = self.pca.transform(X_scaled)
            
            # Predict
            pred = self.clf.predict(X_pca)[0]
            proba = self.clf.predict_proba(X_pca)[0]
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            label = f"Subject {pred} ({max(proba)*100:.1f}%)"
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            recognized_names.append(f"Subject {pred}")
        
        # Display result
        height, width = result_img.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        
        self.result_scene.clear()
        self.result_scene.addItem(QGraphicsPixmapItem(q_pixmap))
        
        # Update recognized faces label
        self.recognized_text.setText(", ".join(recognized_names))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec())