import os
from datetime import datetime

# # Tạo thư mục lưu trữ với timestamp
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# base_save_dir = f"/content/drive/MyDrive/code/ocr/Multi_Type_TD_TSR/output/test_table/test_{timestamp}"

# # Tạo thư mục nếu chưa tồn tại
# if not os.path.exists(base_save_dir):
#     os.makedirs(base_save_dir)
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

def recognize_structure(img, save_dir):
    img_save = os.path.join(save_dir, "img.jpg")
    cv2.imwrite(img_save, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    # img_save = os.path.join(save_dir, "img_bin.jpg")
    # cv2.imwrite(img_save, img_bin)

    img_bin = 255 - img_bin

    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=2)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    # img_save = os.path.join(save_dir, "vertical_lines.jpg")
    # cv2.imwrite(img_save, vertical_lines)

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=2)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    # img_save = os.path.join(save_dir, "horizontal_lines.jpg")
    # cv2.imwrite(img_save, horizontal_lines)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # img_save = os.path.join(save_dir, "combined_lines.jpg")
    # cv2.imwrite(img_save, img_vh)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_vh = cv2.dilate(img_vh, small_kernel, iterations=1)

    edges = cv2.Canny(img_vh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=30)

    img_hough = np.copy(img_vh)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # img_save = os.path.join(save_dir, "hough_lines_detected.jpg")
    # cv2.imwrite(img_save, img_hough)

    bitxor = cv2.bitwise_xor(img, img_hough)
    bitnot = cv2.bitwise_not(bitxor)
    # img_save = os.path.join(save_dir, "bitnot_image.jpg")
    # cv2.imwrite(img_save, bitnot)

    edges_bitnot = cv2.Canny(bitnot, 50, 150, apertureSize=3)
    lines_bitnot = cv2.HoughLinesP(edges_bitnot, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=30)

    img_bitnot_hough = np.copy(bitnot)
    if lines_bitnot is not None:
        for line in lines_bitnot:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_bitnot_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # img_save = os.path.join(save_dir, "bitnot_hough_lines_detected.jpg")
    # cv2.imwrite(img_save, img_bitnot_hough)

    contours, hierarchy = cv2.findContours(img_hough, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    # img_save = os.path.join(save_dir, "contours_detected.jpg")
    # cv2.imwrite(img_save, image)

    return box, img_bin

def create_mask_for_tables(img, boxes, min_width=120, min_height=120):
    """
    Tạo mặt nạ cho các bảng đã được trích xuất nếu bảng đủ điều kiện.
    """
    mask = np.zeros_like(img, dtype=np.uint8)

    for (x, y, w, h) in boxes:
        if w > min_width and h > min_height:
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)

    return mask

def remove_tables_from_image(img, mask):
    """
    Loại bỏ các bảng khỏi ảnh gốc bằng cách sử dụng mặt nạ.
    """
    mask_inv = cv2.bitwise_not(mask)
    img_no_tables = cv2.bitwise_and(img, mask_inv)

    return img_no_tables

def extract_and_save_tables(img, boxes, base_save_dir, min_width=120, min_height=120):
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    mask = create_mask_for_tables(img, boxes)

    img_mask = np.copy(img)

    cv2.imwrite(os.path.join(base_save_dir, "img_mask.jpg"), mask)

    img_no_tables = remove_tables_from_image(img, mask)
    cv2.imwrite(os.path.join(base_save_dir, "img_no_tables.jpg"), img_no_tables)

    coords_file = os.path.join(base_save_dir, "table_coords.txt")

    with open(coords_file, "w") as f:
        for i, (x, y, w, h) in enumerate(boxes):
            table_img = img[y:y+h, x:x+w]
            if w > min_width and h > min_height:
                table_filename = os.path.join(base_save_dir, f"table_{i+1}.jpg")
                cv2.imwrite(table_filename, table_img)
                print(f"Đã lưu bảng {i+1} vào {table_filename}")

                f.write(f"table_{i+1}.jpg {x} {y} {w} {h}\n")
            else:
                print(f"Bảng {i+1} có kích thước quá nhỏ và không được lưu.")

    print(f"Tọa độ của các bảng đã được lưu vào {coords_file}")

# def extract_and_save_tables(img, boxes, base_save_dir, min_width=100, min_height=100):
#     if not os.path.exists(base_save_dir):
#         os.makedirs(base_save_dir)

#     coords_file = os.path.join(base_save_dir, "table_coords.txt")

#     with open(coords_file, "w") as f:
#         for i, (x, y, w, h) in enumerate(boxes):
#             table_img = img[y:y+h, x:x+w]
#             if w > min_width and h > min_height:
#                 table_filename = os.path.join(base_save_dir, f"table_{i+1}.jpg")
#                 cv2.imwrite(table_filename, table_img)
#                 print(f"Đã lưu bảng {i+1} vào {table_filename}")

#                 # Lưu tọa độ của table vào file text
#                 f.write(f"table_{i+1}.jpg {x} {y} {w} {h}\n")
#             else:
#                 print(f"Bảng {i+1} có kích thước quá nhỏ và không được lưu.")

#     print(f"Tọa độ của các bảng đã được lưu vào {coords_file}")


