import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()  # Ẩn cửa sổ chính
image_path = askopenfilename()  # Mở cửa sổ chọn file ảnh

# Đọc ảnh từ đường dẫn đã chọn
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Tính toán Sobel theo hướng x và y
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Kết hợp kết quả Sobel để lấy độ lớn gradient
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Áp dụng Gaussian blur để giảm nhiễu
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Áp dụng toán tử Laplace lên ảnh đã làm mờ
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Dò biên Sobel')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplace Gaussian')
plt.axis('off')

plt.tight_layout()
plt.show()
