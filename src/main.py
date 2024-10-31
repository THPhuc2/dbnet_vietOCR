import cv2
import os
from datetime import datetime
from VietOCR_model import VietOCR_model
from extract_table import recognize_structure, extract_and_save_tables
from ocr_recognition import recognize_text_in_folder
from utils import reassemble_tables_with_recognized_text_to_img


def main(df):
    # Duyệt qua các tệp trong thư mục
    for file_name in os.listdir(df):
        if file_name.endswith(".png"):  # Chỉ xử lý các file .png
            file_path = os.path.join(df, file_name)  # Ghép đường dẫn chính xác

            # Đọc ảnh
            bordered_table = cv2.imread(file_path)

            # Kiểm tra nếu không đọc được ảnh
            if bordered_table is None:
                print(f"Không thể đọc được tệp: {file_path}")
                continue  # Bỏ qua tệp này và xử lý tệp tiếp theo

            # Chuyển sang RGB
            bordered_table = cv2.cvtColor(bordered_table, cv2.COLOR_BGR2RGB)

            # Tạo timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_save_dir = os.path.join(df, f"test_{os.path.splitext(file_name)[0]}_{timestamp}")

            # Tạo thư mục nếu chưa có
            if not os.path.exists(base_save_dir):
                os.makedirs(base_save_dir)

            list_table_boxes = []
            table_list = [bordered_table]

            # Xử lý từng bảng trong danh sách table_list
            for table in table_list:
                finalboxes, output_img = recognize_structure(table, base_save_dir)
                list_table_boxes.append(finalboxes)
                extract_and_save_tables(bordered_table, finalboxes, base_save_dir)

            # Nhận diện văn bản trong thư mục và tái tạo bảng
            recognize_text_in_folder(base_save_dir, VietOCR_model, base_save_dir)
            reassemble_tables_with_recognized_text_to_img(base_save_dir, base_save_dir)

if __name__ == "__main__":
    df = input("Nhập đường dẫn thư mục chứa ảnh (có thể dùng dấu / hoặc \\) ảnh là .png không nhập ảnh khác: ")
    main(df)
