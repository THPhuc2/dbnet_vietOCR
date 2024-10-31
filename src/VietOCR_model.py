from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def load_vietocr(device="cuda:0", name='vgg_transformer', weight_path='D:\\CODE\\Deep_Learning\\Project\\OCR_dbnet_craft_vietocr\\vietocr\\weights\\Vietocr\\transformerocr_last.pth'):
    config = Cfg.load_config_from_name(name)
    config['weights'] = weight_path
    config['cnn']['pretrained'] = False
    config['device'] = device

    return Predictor(config)

def VietOCR_model(device="cuda:0"):
    return load_vietocr(device=device)