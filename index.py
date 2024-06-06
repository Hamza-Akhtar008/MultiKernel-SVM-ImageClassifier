import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn import svm
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import mahotas
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function to save and load features
def save_features(features, file_path):
    np.save(file_path, features)


def load_features(file_path):
    return np.load(file_path)


# Function to load images from a folder and assign labels
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    labels = []

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)

                if "airplane" in filename.lower():
                    labels.append("airplane")
                elif "car" in filename.lower():
                    labels.append("car")
                elif "cat" in filename.lower():
                    labels.append("cat")
                else:
                    continue
    return np.array(images), np.array(labels)


# Function to augment images
def augment_images(images, labels, augmentation_factor=2, target_size=(128, 128)):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        for _ in range(augmentation_factor):
            augmented_img = img.copy()

            angle = np.random.randint(-15, 15)
            rows, cols = augmented_img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, (cols, rows))

            if np.random.choice([True, False]):
                augmented_img = cv2.flip(augmented_img, 1)


            augmented_img = cv2.resize(augmented_img, target_size, interpolation=cv2.INTER_LINEAR)

            augmented_images.append(augmented_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        denoised_img = cv2.GaussianBlur(equalized_img, (3, 3), 0)
        preprocessed_images.append(denoised_img)
    return np.array(preprocessed_images)


def plot_decision_boundaries(X, y, classifier, title, step_size=0.5):
    # Create a mesh grid with a reduced step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # Predict in batches to avoid memory issues
    ravel_shape = xx.ravel().shape[0]
    batch_size = 100000  # Number of points to predict at once
    Z = np.zeros(ravel_shape)

    for i in range(0, ravel_shape, batch_size):
        end = i + batch_size
        if end > ravel_shape:
            end = ravel_shape
        Z[i:end] = classifier.predict(np.c_[xx.ravel()[i:end], yy.ravel()[i:end]])

    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
def extract_hog_features(images, cell_size=(8, 8), block_size=(2, 2), bins=9):
    hog_features = []
    hog = cv2.HOGDescriptor(_winSize=(images.shape[2] // cell_size[1] * cell_size[1],
                                      images.shape[1] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=bins)
    for img in images:
        if img is None:
            print("Warning: Empty image encountered, skipping...")
            continue

        img_resized = cv2.resize(img, (128, 128))
        hog_feature = hog.compute(img_resized)
        hog_features.append(hog_feature.flatten())
    return np.array(hog_features)


def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        lbp = mahotas.features.lbp(img, 24, 8, True)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        lbp_features.append(lbp_hist)
    return np.array(lbp_features)


def extract_deep_cnn_features(images):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = []
    for img in images:
        img_resized = cv2.resize(img, (224, 224))
        img_preprocessed = preprocess_input(img_resized)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        feature = model.predict(img_expanded)
        features.append(feature.flatten())
    return np.array(features)





def train_svm(features, labels, kernel='linear'):
    classifier = svm.SVC(kernel=kernel)
    classifier.fit(features, labels)
    return classifier


def evaluate_classifier(classifier, features, true_labels, label_mapping):
    predictions = classifier.predict(features)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    predicted_labels = [label_mapping[pred] for pred in predictions]
    return predicted_labels
def plot_metrics(kernel, accuracy, precision, recall, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title(f'Model Performance Metrics for {kernel} Kernel')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.show()

def display_image_with_prediction(kernel,image_path, predicted_class):
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with the predicted class
    plt.imshow(img_rgb)
    plt.title(f'Predicted class by SVM {kernel} :    {predicted_class}')

    plt.axis('off')
    plt.show()

def predict_single_image(image, classifier, pca, label_mapping):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    denoised_img = cv2.GaussianBlur(equalized_img, (3, 3), 0)

    img_resized = cv2.resize(denoised_img, (128, 128))

    hog_feature = extract_hog_features(np.array([img_resized]))
    lbp_feature = extract_lbp_features(np.array([img_resized]))
    cnn_feature = extract_deep_cnn_features(np.array([image]))

    if hog_feature.size == 0:
        features = np.concatenate((lbp_feature, cnn_feature), axis=1)
    else:
        features = np.concatenate((hog_feature, lbp_feature, cnn_feature), axis=1)

        pca_feature = pca.transform(features)

        prediction = classifier.predict(pca_feature)
        class_name = label_mapping[prediction[0]]
        return class_name


def main():
    images, labels = load_images_from_folder("C:/Users/Administrator/Desktop/dataset")

    label_mapping = {"airplane": 0, "car": 1, "cat": 2}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    numeric_labels = np.array([label_mapping[label] for label in labels])

    augmented_images, augmented_labels = augment_images(images, numeric_labels)
    preprocessed_images = preprocess_images(augmented_images)

    # Check if features exist, load them if available, otherwise extract and save them
    hog_file = 'hog_features.npy'
    lbp_file = 'lbp_features.npy'
    cnn_file = 'cnn_features.npy'

    if os.path.isfile(hog_file):
        hog_features = load_features(hog_file)
    else:
        hog_features = extract_hog_features(preprocessed_images)
        save_features(hog_features, hog_file)

    if os.path.isfile(lbp_file):
        lbp_features = load_features(lbp_file)
    else:
        lbp_features = extract_lbp_features(preprocessed_images)
        save_features(lbp_features, lbp_file)

    if os.path.isfile(cnn_file):
        cnn_features = load_features(cnn_file)
    else:
        cnn_features = extract_deep_cnn_features(augmented_images)
        save_features(cnn_features, cnn_file)

    all_features = np.concatenate((hog_features, lbp_features, cnn_features), axis=1)
    pca = PCA(n_components=100)
    pca_features = pca.fit_transform(all_features)
    classifier = train_svm(pca_features, augmented_labels)

    X_train, X_test, y_train, y_test = train_test_split(pca_features, augmented_labels, test_size=0.2,
                                                        random_state=42)
    # Train SVM classifiers with different kernels
    classifier_linear = train_svm(X_train, y_train, kernel='linear')
    classifier_poly = train_svm(X_train, y_train, kernel='poly')
    classifier_rbf = train_svm(X_train, y_train, kernel='rbf')

    # Evaluate classifiers
    evaluate_classifier(classifier_linear, X_test, y_test, reverse_label_mapping)
    evaluate_classifier(classifier_poly, X_test, y_test, reverse_label_mapping)
    evaluate_classifier(classifier_rbf, X_test, y_test, reverse_label_mapping)

    # Calculate metrics for each classifier
    accuracy_linear = accuracy_score(y_test, classifier_linear.predict(X_test))
    precision_linear = precision_score(y_test, classifier_linear.predict(X_test), average='weighted', zero_division=0)
    recall_linear = recall_score(y_test, classifier_linear.predict(X_test), average='weighted', zero_division=0)
    f1_linear = f1_score(y_test, classifier_linear.predict(X_test), average='weighted', zero_division=0)

    accuracy_poly = accuracy_score(y_test, classifier_poly.predict(X_test))
    precision_poly = precision_score(y_test, classifier_poly.predict(X_test), average='weighted', zero_division=0)
    recall_poly = recall_score(y_test, classifier_poly.predict(X_test), average='weighted', zero_division=0)
    f1_poly = f1_score(y_test, classifier_poly.predict(X_test), average='weighted', zero_division=0)

    accuracy_rbf = accuracy_score(y_test, classifier_rbf.predict(X_test))
    precision_rbf = precision_score(y_test, classifier_rbf.predict(X_test), average='weighted', zero_division=0)
    recall_rbf = recall_score(y_test, classifier_rbf.predict(X_test), average='weighted', zero_division=0)
    f1_rbf = f1_score(y_test, classifier_rbf.predict(X_test), average='weighted', zero_division=0)

    test_image_path = "C:/Users/Administrator/Desktop/test_image.jpg"
    test_image = cv2.imread(test_image_path)

    # Predict single image for each classifier
    if test_image is not None:
        predicted_class_linear = predict_single_image(test_image, classifier_linear, pca, reverse_label_mapping)
        predicted_class_poly = predict_single_image(test_image, classifier_poly, pca, reverse_label_mapping)
        predicted_class_rbf = predict_single_image(test_image, classifier_rbf, pca, reverse_label_mapping)
        print(f"Linear kernel predicted class: {predicted_class_linear}")
        print(f"Polynomial kernel predicted class: {predicted_class_poly}")
        print(f"RBF kernel predicted class: {predicted_class_rbf}")
        display_image_with_prediction('linear',test_image_path, predicted_class_linear)
        display_image_with_prediction('poly',test_image_path, predicted_class_poly)
        display_image_with_prediction('rbf',test_image_path, predicted_class_rbf)
    else:
        print("Error: Could not read test image.")

    # Plot metrics for each classifier
    plot_metrics('linear',accuracy_linear, precision_linear, recall_linear, f1_linear)
    plot_metrics('poly',accuracy_poly, precision_poly, recall_poly, f1_poly)
    plot_metrics('rbf',accuracy_rbf, precision_rbf, recall_rbf, f1_rbf)

    # Plot decision boundaries for the first two principal components
    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    classifier_linear_2d = train_svm(X_train_2d, y_train, kernel='linear')
    classifier_poly_2d = train_svm(X_train_2d, y_train, kernel='poly')
    classifier_rbf_2d = train_svm(X_train_2d, y_train, kernel='rbf')

    plot_decision_boundaries(X_test_2d, y_test, classifier_linear_2d, 'SVM Decision Boundary with Linear Kernel')
    plot_decision_boundaries(X_test_2d, y_test, classifier_poly_2d, 'SVM Decision Boundary with Polynomial Kernel')
    plot_decision_boundaries(X_test_2d, y_test, classifier_rbf_2d, 'SVM Decision Boundary with RBF Kernel')


if __name__ == "__main__":
    main()

