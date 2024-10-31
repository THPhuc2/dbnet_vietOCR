# %cd /content/drive/MyDrive/code/ocr/text_detection/src/DB_text_minimal/src
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from models import DBTextModel
from utils import test_preprocess
from postprocess import SegDetectorRepresenter
import os
import torch
import cv2
import numpy as np
import gc
# from google.colab.patches import cv2_imshow

# Khởi tạo mô hình DBNet  dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth
def dbnet(model_path="../../../weights/best_dbnet.pth", device="cpu"):
    """
    Tải mô hình DBNet.

    :param model_path: Đường dẫn đến mô hình.
    :param device: Thiết bị (cpu hoặc cuda).
    :return: Đối tượng mô hình DBNet đã tải.
    """
    assert os.path.exists(model_path), "Model path does not exist."
    dbnet_model = DBTextModel().to(device)
    dbnet_model.load_state_dict(torch.load(model_path, map_location=device))
    dbnet_model.eval()
    return dbnet_model

# Các hàm phụ trợ
def preprocess(image_arr, device="cpu"):
    """
    Tiền xử lý hình ảnh đầu vào.

    :param image_arr: Mảng hình ảnh đầu vào.
    :param device: Thiết bị (cpu hoặc cuda).
    :return: Hình ảnh đã được tiền xử lý.
    """
    preprocessed_image = test_preprocess(image_arr, to_tensor=True, pad=False).to(device)
    return preprocessed_image

def get_detected_bbox(image_arr, is_output_polygon=False):
    """
    Nhận diện bounding boxes từ hình ảnh.

    :param image_arr: Mảng hình ảnh đầu vào.
    :param is_output_polygon: Tham số để quyết định đầu ra là polygon hay không.
    :return: Danh sách các bounding box đã phát hiện.
    """
    global dbnet
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB)
    preprocessed_image = preprocess(image_arr)
    h_origin, w_origin, _ = image_arr.shape

    with torch.no_grad():
        preds = dbnet(preprocessed_image)

    seg_obj = SegDetectorRepresenter(thresh=0.1, box_thresh=0.4, unclip_ratio=1.5)
    batch = {'shape': [(h_origin, w_origin)]}
    box_list, _ = seg_obj(batch, preds, is_output_polygon=is_output_polygon)
    box_list = box_list[0]

    if len(box_list) > 0:
        if is_output_polygon:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list = box_list[idx]
    else:
        box_list = []

    return np.array(box_list)

def draw_bbox(img, bboxes, color=(255, 0, 0), thickness=1):
    """
    Vẽ bounding box lên hình ảnh.

    :param img: Hình ảnh đầu vào (có thể là đường dẫn hoặc mảng hình ảnh).
    :param bboxes: Danh sách các bounding box.
    :param color: Màu sắc cho bounding box (mặc định là đỏ).
    :param thickness: Độ dày của bounding box.
    :return: Hình ảnh đã vẽ bounding box.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for bbox in bboxes:
        bbox = bbox.astype(int)
        cv2.polylines(img, [bbox], True, color, thickness)
    return img
def detect(image_arr, device="cpu"):
    """
    Phát hiện văn bản trong hình ảnh.

    :param image_arr: Mảng hình ảnh đầu vào.
    :param device: Thiết bị (cpu hoặc cuda).
    :return: Danh sách bounding box và hình ảnh đã vẽ bounding box.
    """
    bboxes = get_detected_bbox(image_arr)
    detected_image = draw_bbox(image_arr, bboxes)

    return bboxes, detected_image

def get_min_max_pos(bbox):
    """
    Lấy vị trí tối thiểu và tối đa của bounding box.

    :param bbox: Mảng bounding box.
    :return: (minx, miny, maxx, maxy) vị trí của bounding box.
    """
    minx = min(bbox[:, 0])
    maxx = max(bbox[:, 0])
    miny = min(bbox[:, 1])
    maxy = max(bbox[:, 1])

    return minx, miny, maxx, maxy

def get_cropped_area(image_arr, bbox):
    """
    Cắt vùng hình ảnh dựa trên bounding box.

    :param image_arr: Mảng hình ảnh đầu vào.
    :param bbox: Mảng bounding box.
    :return: Hình ảnh đã cắt.
    """
    minx, miny, maxx, maxy = get_min_max_pos(bbox)

    return image_arr[miny:maxy, minx:maxx]

# def recog(image_arr, bboxes, device="cuda"):
#     global model

#     torch.cuda.empty_cache()
#     gc.collect()

#     cropped_image_batch = [Image.fromarray(get_cropped_area(image_arr, bbox)) for bbox in bboxes]
#     pred_texts = model.predict_batch(cropped_image_batch[-1::-1])

#     return pred_texts

# Nhận diện và vẽ kết quả văn bản lên hình ảnh
def recog_and_draw(image_arr, bboxes, model, font_path="/content/drive/MyDrive/code/ocr/vietocr/Arial.ttf", device="cpu"):
    """
    Nhận diện văn bản và vẽ kết quả lên hình ảnh.

    :param image_arr: Mảng hình ảnh đầu vào.
    :param bboxes: Danh sách bounding box đã phát hiện.
    :param model: Mô hình OCR để nhận diện văn bản.
    :param font_path: Đường dẫn đến font để hiển thị văn bản.
    :param device: Thiết bị (cpu hoặc cuda).
    :return: Hình ảnh đã vẽ kết quả văn bản.
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Sử dụng font hỗ trợ tiếng Việt
    try:
        font = ImageFont.truetype(font_path, 16)  # Kích thước font là 20, bạn có thể điều chỉnh
    except IOError:
        print(f"Không tìm thấy font tại {font_path}. Sử dụng font mặc định.")
        font = ImageFont.load_default()

    pil_image = Image.fromarray(image_arr)
    draw = ImageDraw.Draw(pil_image)

    # Nhận diện văn bản cho từng vùng cắt
    for bbox in bboxes:
        cropped_image = Image.fromarray(get_cropped_area(image_arr, bbox))
        pred_text = model.predict(cropped_image)

        # Vẽ bounding box và chèn text lên hình ảnh
        minx, miny, maxx, maxy = get_min_max_pos(bbox)
        draw.rectangle([minx, miny, maxx, maxy], outline="red", width=2)
        draw.text((minx, miny - 10), pred_text, fill="red", font=font)

    return np.array(pil_image)




# def create_result(image_arr, device="cpu"):
#     import time
#     start = time.time()
#     bboxes, detected_image = detect(image_arr, device)
#     detect_time = time.time() - start

#     start = time.time()
#     pred_texts = recog(detected_image, bboxes, device)
#     recog_time = time.time() - start

#     return detected_image, "\n".join(pred_texts)


# Hàm chạy toàn bộ quá trình từ phát hiện đến nhận diện và vẽ kết quả
def create_result(image_arr, ocr_model, device="cpu"):
    # Phát hiện bounding boxes
    bboxes, detected_image = detect(image_arr, device)

    # Nhận diện văn bản và vẽ kết quả lên hình
    result_image = recog_and_draw(image_arr, bboxes, ocr_model, device=device)

    return result_image
