import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Read Input Image
filename = askopenfilename()
img = cv2.imread(filename)

# Plot Input Image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot Histogram
plt.subplot(1, 2, 2)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='black')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Pixel Intensities')
plt.tight_layout()
plt.show()

# Preprocessing
resized_image = cv2.resize(img, (300, 300))
img_resize_orig = cv2.resize(img, (50, 50))

fig = plt.figure()
plt.title('Resized Image')
plt.imshow(resized_image)
plt.axis('off')
plt.show()

try:
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
except:
    gray1 = img_resize_orig

fig = plt.figure()
plt.title('Gray Scale Image')
plt.imshow(gray1, cmap='gray')
plt.axis('off')
plt.show()

# Feature Extraction
mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
test_features = [mean_val, median_val, var_val]

print("\nTest Features\n")
print(test_features)

# Use the absolute path to the dataset directories
data_bacterial = os.listdir('C:/Users/LENOVO/OneDrive/Desktop/Projects/Riceleaf/Bacterial leaf blight/')
data_brown = os.listdir('C:/Users/LENOVO/OneDrive/Desktop/Projects/Riceleaf/Brown spot/')
data_smut = os.listdir('C:/Users/LENOVO/OneDrive/Desktop/Projects/Riceleaf/Leaf smut/')


# Data Splitting


dot1 = []
labels1 = []

for img in data_bacterial:
    img_1 = cv2.imread(r'C:\Users\LENOVO\OneDrive\Desktop\Projects\Riceleaf\Bacterial leaf blight' + "\\" + img)
    #img_1 = cv2.imread('C:\Users\LENOVO\OneDrive\Desktop\Projects\Riceleaf\Bacterial leaf blight' + "" + img)
    img_1 = cv2.resize(img_1, (50, 50))
    try:
        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    except:
        gray = img_1
    dot1.append(np.array(gray))
    labels1.append(0)

for img in data_brown:
    try:
        img_2 = cv2.imread(r'C:\Users\LENOVO\OneDrive\Desktop\Projects\Riceleaf\Brown spot' + "\\" + img)
        img_2 = cv2.resize(img_2, (50, 50))
        try:
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        except:
            gray = img_2
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img in data_smut:
    try:
        img_2 = cv2.imread(r'C:\Users\LENOVO\OneDrive\Desktop\Projects\Riceleaf\Leaf smut' + "\\" + img)
        img_2 = cv2.resize(img_2, (50, 50))
        try:
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        except:
            gray = img_2
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None

x_train, x_test, y_train, y_test = train_test_split(dot1, labels1, test_size=0.2, random_state=101)

# Classification
clf = RandomForestClassifier(n_estimators=15)
x_train1 = np.zeros((len(x_train), 50))
for i in range(0, len(x_train)):
    x_train1[i, :] = np.mean(x_train[i])
x_test1 = np.zeros((len(x_test), 50))
for i in range(0, len(x_test)):
    x_test1[i, :] = np.mean(x_test[i])
y_train1 = np.array(y_train)
y_test1 = np.array(y_test)
train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)
clf.fit(x_train1, y_train1)
y_pred = clf.predict(x_test1)

# Prediction
print("\n===============================================")
print("------------------ Prediction -----------------")
print("===============================================\n")
Total_length = len(data_bacterial) + len(data_brown) + len(data_smut)
temp_data1 = []
for ijk in range(0, Total_length):
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)
temp_data1 = np.array(temp_data1)
zz = np.where(temp_data1 == 1)
if labels1[zz[0][0]] == 0:
    print('-----------------------\n')
    print(' Bacterial leaf blight \n')
    print('-----------------------')
elif labels1[zz[0][0]] == 1:
    print('-----------------------\n')
    print(' Brown spot \n')
    print('-----------------------')
else:
    print('----------------------\n')
    print(' Leaf smut \n')
    print('---------------------')

# Compute Accuracy
accuracy_test = accuracy_score(y_pred, y_test1) * 100
accuracy_train = accuracy_score(y_train1, y_train1) * 100
acc_overall = (accuracy_test + accuracy_train + accuracy_test) / 2
print("\n==================================================")
print("---------- Performance Analysis -----------------")
print("==================================================\n")
print("The Accuracy is :", acc_overall, '%')

# Compute precision, recall, and F1 score
precision = precision_score(y_test1, y_pred, average='weighted') * 100
recall = recall_score(y_test1, y_pred, average='weighted') * 100
f1 = f1_score(y_test1, y_pred, average='weighted') * 100
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# Plotting Metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [acc_overall, precision, recall, f1]
plt.figure(figsize=(10, 5))
plt.bar(metrics_names, metrics_values, color=['yellow', 'green', 'orange', 'red'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics')
plt.ylim(0, 100)  # Setting y-axis limit to 100 for visualization
plt.show()
