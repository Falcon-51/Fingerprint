import gradio as gr
import cv2
import numpy as np
import os

class FingerprintDetector:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.template_features = self.load_templates()

    def preprocess_image(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        return img

    def load_templates(self):
        templates = []
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            img = self.preprocess_image(file_path)
            if img is not None:
                templates.append(img)
        return templates

    def match_fingerprints(self, input_features, threshold=0.9):
        if not self.template_features or input_features is None:
            return "Ошибка загрузки изображений"
        input_features = input_features.astype(np.float32)
        for template in self.template_features:
            template = template.astype(np.float32)
            result = cv2.matchTemplate(input_features, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return "Отпечатки совпадают"
        return "Отпечатки не совпадают"

def fingerprint_match(input_img, folder_path="/Users/sulakovasentyabrina/Magistratura/Fingerprint-main/base"):
    detector = FingerprintDetector(folder_path)
    input_features = detector.preprocess_image(input_img)
    return detector.match_fingerprints(input_features)

gui = gr.Interface(
    fn=fingerprint_match,
    inputs=[gr.Image(type="filepath"), gr.Textbox(label="База с отпечатками", value="/Users/sulakovasentyabrina/Magistratura/Fingerprint-main/base")],
    outputs=gr.Textbox(label="Результат"),
    title="Система Сопоставления Отпечатков Пальцев (ССОП)",
    description="Загрузите изображение отпечатка пальца и укажите папку с отпечатками для сравнения.",
    allow_flagging="never"
)

gui.launch()
