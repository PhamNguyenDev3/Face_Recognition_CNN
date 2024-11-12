
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = 'Datasets/ThuThao/'
filename = 'ThuThao.csv'
files = os.listdir(path)
dim = (100, 100)  # Kích thước mới là 100x100
cls = 2
df = pd.DataFrame(columns=[f'pix-{i}' for i in range(1, 1 + (dim[0] * dim[1]))] + ['class'])

# Lưu ảnh reshape vào một list để in ra
reshaped_images = []

for i in tqdm(range(1, 1 + len(files))):
    img = Image.open(path + files[i - 1])
    
    # Resize ảnh về kích thước mới (100x100)
    resized_img = img.resize(dim)
    
    # Chuyển đổi thành mảng numpy và flatten nó
    flattened_img = np.array(resized_img).flatten()
    
    # Thêm dữ liệu vào dataframe
    df.loc[i] = flattened_img.tolist() + [cls]
    
    # Thêm ảnh đã reshape vào list
    reshaped_images.append(resized_img)

# Lưu dataframe vào file CSV
df.to_csv(filename, index=False)
print('Task Completed')

# In ảnh đã reshape
plt.figure(figsize=(10, 10))
for i in range(9):  # In 9 ảnh đầu tiên
    plt.subplot(3, 3, i + 1)
    plt.imshow(reshaped_images[i], cmap='gray')
    plt.axis('off')
plt.show()
