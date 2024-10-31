import os
import numpy as np
import cv2



# Bước 3: Ghép lại các bảng đã nhận diện vào ảnh có tên là 'img.jpg'
def reassemble_tables_with_recognized_text_to_img(img_folder_path, table_folder_path):
    """
    Ghép các bảng đã nhận diện vào ảnh có tên là 'img.jpg' trong thư mục.
    """
    coords_file = os.path.join(table_folder_path, "table_coords.txt")

    # Đọc file chứa tọa độ của các bảng
    with open(coords_file, "r") as f:
        lines = f.readlines()

    img_path = os.path.join(img_folder_path, "img_no_tables.jpg")  # Đường dẫn tới ảnh có tên 'img.jpg' img_recognized.jpg
    img = cv2.imread(img_path)

    # Ghép lại các bảng đã nhận diện vào vị trí ban đầu
    for line in lines:
        table_file, x, y, w, h = line.split()
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Đọc lại bảng đã nhận diện (result_table_x.jpg)
        result_table_path = os.path.join(table_folder_path, table_file.replace("table_", "result_table_"))
        result_table_img = cv2.imread(result_table_path)

        # Ghép lại bảng vào ảnh 'img.jpg'
        img[y:y+h, x:x+w] = result_table_img
        print(f"Ghép lại bảng {result_table_path} vào vị trí ({x}, {y}, {w}, {h}) trên ảnh img.jpg")

    # Lưu ảnh sau khi ghép lại
    result_img_path = os.path.join(img_folder_path, "final_img_with_tables.jpg")
    cv2.imwrite(result_img_path, img)
    print(f"Ảnh img_recognized.jpg đã được ghép lại và lưu tại: {result_img_path}")