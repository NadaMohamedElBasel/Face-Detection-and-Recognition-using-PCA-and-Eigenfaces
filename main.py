from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QRadioButton, QSpinBox,
                             QGraphicsView, QGraphicsScene, QFrame,QFileDialog,QGraphicsPixmapItem)
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc ############ need check if using sklearn is allowed or not ? #################
import matplotlib.pyplot as plt
class FaceRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 700)
        
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
        self.apply_pca_btn = QPushButton("Apply PCA")
        self.apply_pca_btn.clicked.connect(self.apply_pca)
        self.recognize_faces_btn = QPushButton("Recognize Faces")
        self.recognize_faces_btn.clicked.connect(self.recognize_faces)
        
        # Face Recognition Result Display
        self.recognized_label = QLabel("Recognized Faces: ")
        self.recognized_text = QLabel("N/A")
        
        # Radio Buttons for Color/Grayscale
        self.color_radio = QRadioButton("Color")
        self.grayscale_radio = QRadioButton("Grayscale")
        self.grayscale_radio.setChecked(True)
        
        # PCA Components Selector
        self.pca_spinbox = QSpinBox()
        self.pca_spinbox.setRange(1, 100)
        self.pca_spinbox.setValue(10)
        
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
        left_panel.addWidget(self.apply_pca_btn)
        left_panel.addWidget(self.recognize_faces_btn)
        left_panel.addWidget(self.recognized_label)
        left_panel.addWidget(self.recognized_text)
        left_panel.addWidget(self.color_radio)
        left_panel.addWidget(self.grayscale_radio)
        left_panel.addWidget(QLabel("Number of PCA Components:"))
        left_panel.addWidget(self.pca_spinbox)
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

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        img = cv2.imread(file_path)
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

        # Convert OpenCV image (NumPy array) to QImage
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert QImage to QPixmap
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.input_view.setScene(scene)

    def detect_faces(self):
        pass 
        # color or gray from the radio button?
        #call update_performance_metrics 

    def apply_pca(self):
        pass
        # take number of PCA components from the spinbox
        #call update_performance_metrics

    def recognize_faces(self):
        pass
        # color or gray from the radio button?
        #call update_performance_metrics



    ######### assuming y-pred , y_scores , y_true are output from detection / recognition ##########

    def calculate_performance_metrics(self,y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall

    def plot_roc_curve(self,y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig("roc_curve.png")
        plt.close()
        return "roc_curve.png"
    
    def update_performance_metrics(self, y_true, y_pred, y_scores): # to be called inside functions of detection and recognition
        accuracy, precision, recall = self.calculate_performance_metrics(y_true, y_pred)
        self.performance_value.setText(f"{accuracy:.2f}")
        self.precision_value.setText(f"{precision:.2f}")
        self.recall_value.setText(f"{recall:.2f}")
        
        roc_img_path = self.plot_roc_curve(y_true, y_scores)
        pixmap = QPixmap(roc_img_path)
        self.roc_scene.clear()
        self.roc_scene.addItem(QGraphicsPixmapItem(pixmap))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec())