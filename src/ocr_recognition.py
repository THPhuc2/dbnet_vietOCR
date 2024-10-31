import os
import cv2
import numpy as np

def recognize_text_in_folder(folder_path, model, result_save_dir):
    """
    Thực hiện nhận diện văn bản cho tất cả các bảng trong folder và lưu kết quả.
    """
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".jpg"):
            # Đọc ảnh
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Thực hiện nhận diện văn bản
            result_image = create_result(image, model, device="cpu")  # Gọi hàm nhận diện văn bản

            # Xác định tên ảnh kết quả
            if filename == "img.jpg":
                result_filename = "img_recognized.jpg"
            else:
                result_filename = filename.replace("table_", "result_table_")

            # Lưu kết quả nhận diện với tên phù hợp
            result_path = os.path.join(result_save_dir, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"Đã lưu kết quả nhận diện: {result_path}")
