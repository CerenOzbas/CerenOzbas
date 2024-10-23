# Akbank Derin Öğrenme Projesi

Bu proje, Akbank tarafından sağlanan veri seti üzerinde **derin öğrenme** tekniklerinin uygulanmasını kapsamaktadır. Projede, Python programlama dili ve popüler yapay zeka kütüphaneleri kullanılarak veri analizi ve model geliştirme çalışmaları gerçekleştirilmiştir.

## Kaggle Projesi
Projeye Kaggle üzerinden [buradan ulaşabilirsiniz](https://www.kaggle.com/code/ceren23/akbank-derin-renme-ceren-zba).

## Açıklama
Bu çalışmada:
- Akbank verileri üzerinde veri temizleme ve ön işleme teknikleri uygulandı.
- Derin öğrenme modelleri geliştirildi ve performansları değerlendirildi.
- Eğitim ve test sonuçları görselleştirildi.

## Kullanılan Teknolojiler
- **Python 3.8+**
- **Keras** ve **TensorFlow**: Derin öğrenme için
- **Pandas** ve **NumPy**: Veri işleme için
- **Matplotlib** ve **Seaborn**: Görselleştirme için

## Gerekli Kütüphaneler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
## Veri Setinin Kütüphaneye Eklenmesi
data_dir = '../input/a-large-scale-fish-dataset/Fish_Dataset/'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  
train_data = datagen.flow_from_directory(data_dir, 
                                         target_size=(224, 224), 
                                         batch_size=32, 
                                         class_mode='categorical', 
                                         subset='training')  

val_data = datagen.flow_from_directory(data_dir, 
                                       target_size=(224, 224), 
                                       batch_size=32, 
                                       class_mode='categorical', 
                                       subset='validation')  
Found 14400 images belonging to 1 classes.
Found 3600 images belonging to 1 classes.
## Veri Sayısı
data_dir = '../input/a-large-scale-fish-dataset/Fish_Dataset/'

def count_images_in_directory(directory):
    total_images = 0
    for root, dirs, files in os.walk(directory):
        total_images += len([file for file in files if file.endswith('.png')])
    return total_images

## Gözlem sayısı hesaplanması
total_images = count_images_in_directory(data_dir)
print(f'Toplam gözlem (görüntü) sayısı: {total_images}')
Toplam gözlem (görüntü) sayısı: 18000

## Train ve Test Setlerinin Boyutları

train_size = train_data.samples
print(f"Eğitim seti boyutu: {train_size} görüntü")

val_size = val_data.samples
print(f"Doğrulama seti boyutu: {val_size} görüntü")
Eğitim seti boyutu: 14400 görüntü
Doğrulama seti boyutu: 3600 görüntü

## Görselletirme
sample_training_images, _ = next(train_data)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:10]) #ilk 10 balık resmi görselleştirilir.

## Piksel Yoğunluğu İncelemesi 

plt.figure(figsize=(8, 6))
plt.hist(sample_training_images.flatten(), bins=50, color='red', alpha=0.7)  
         
plt.title("Tüm Piksel Değerlerinin Dağılımı (0 - 1 aralığında)")
plt.xlabel("Piksel Değeri")
plt.ylabel("Frekans") 
plt.show()
## Tür Sınıfları
DIR = '/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset' 
classes = [i for i in os.listdir(DIR) if '.' not in i]                     
classes
['Hourse Mackerel',
 'Black Sea Sprat',
 'Sea Bass',
 'Red Mullet',
 'Trout',
 'Striped Red Mullet',
 'Shrimp',
 'Gilt-Head Bream',
 'Red Sea Bream']
 ## Tür Görselleri
 label = []
path = []
​
for dirname, _,filenames in os.walk(DIR):                    
    for filename in filenames:                                 
        if os.path.splitext(filename)[-1]=='.png':              
            if dirname.split()[-1]!='GT':                       
                label.append(os.path.split(dirname)[-1])         
                path.append(os.path.join(dirname,filename))     
​
df = pd.DataFrame(columns=['path','label'])
df['path']=path
df['label']=label
df.info()
df['label'].value_counts()
idx = 0
plt.figure(figsize=(15,12))
for unique_label in df['label'].unique():
    plt.subplot(3, 3, idx+1)
    plt.imshow(plt.imread(df[df['label']==unique_label].iloc[0,0]))
    plt.title(unique_label)
    plt.axis('off')
    idx+=1
    <class 'pandas.core.frame.DataFrame'>
RangeIndex: 9000 entries, 0 to 8999
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   path    9000 non-null   object
 1   label   9000 non-null   object
dtypes: object(2)
memory usage: 140.8+ KB

## Train test
train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
print(train_df.shape)
print(test_df.shape)
(7200, 2)
(1800, 2)

## Encoding
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_images = train_generator.flow_from_dataframe(dataframe=train_df, x_col='path', y_col='label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=True, seed=42, subset='training')
val_images = train_generator.flow_from_dataframe(dataframe=train_df, x_col='path', y_col='label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=True, seed=42, subset='validation' )
test_images = test_generator.flow_from_dataframe(dataframe=test_df, x_col='path', y_col='label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=False )

Found 5760 validated image filenames belonging to 9 classes.
Found 1440 validated image filenames belonging to 9 classes.
Found 1800 validated image filenames belonging to 9 classes.

## 20 görselin 0 - 1 atamaları
import matplotlib.pyplot as plt


fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,12))
ax = ax.flatten()
for j in range(20):
    img, label = next(test_images)  
   
    if isinstance(label, (list, np.ndarray)):
        label = label[0]
    
    if not isinstance(label, str):
        label = str(label)
    
    ax[j].imshow(img[0])  
    ax[j].set_title(label) 
    ax[j].axis('off')  

plt.tight_layout()  
plt.show()  

## Encoding
print(df.columns)
import pandas as pd
X = df['path']  
y = df['label']  # Balık türleri

from sklearn.preprocessing import LabelEncoder

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(list(zip(le.classes_, range(len(le.classes_)))))

## Normalizasyon
import cv2
import numpy as np

# Görüntüleri yükleme ve boyutlandırma (örneğin 128x128)
images = [cv2.resize(cv2.imread(img_path), (128, 128)) for img_path in X]

# Görüntüleri NumPy dizisine dönüştürme
X_images = np.array(images) / 255.0  # 0-1 aralığında normalizasyon
from sklearn.model_selection import train_test_split

# Veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X_images, y_encoded, test_size=0.2, random_state=42)

print(f"Eğitim veri boyutu: {X_train.shape}, Test veri boyutu: {X_test.shape}")


## Loss grafiği
plt.subplot(1, 2, 1)
plt.plot(results.history['loss'], label='Train Loss')
plt.plot(results.history['val_loss'], label='Validation Loss')
plt.title('Loss Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## Accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(results.history['accuracy'], label='Train Accuracy')
plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
































