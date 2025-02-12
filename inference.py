from Fingerprint.fingerprint import FingerprintDetector
import gradio as gr  
import os
from PIL import Image, ImageDraw, ImageFont
import time


def infer(func:FingerprintDetector=1) -> None:
    """
    Запускает веб-приложение

    Параметры: None

    Возвращает: None
    """

    # Определяем веб-интерфейс Gradio для взаимодействия с моделью YOLO.
    iface = gr.Interface(
        fn=time.sleep(1),  # Указываем функцию предсказания, которая будет вызываться при загрузке изображения.
        inputs=[
            gr.Image(type="pil", label="Загруженное изображение"),  # Задаем тип входных данных (изображение формата PIL).
        ],
        
        outputs=[
            gr.Image(type="pil", label="Результат"),  # Первое выходное значение - это изображение
            gr.Textbox(label="Вывод")  # Второе выходное значение - список объектов
        ], 
        title="Fingerprint detector",  
        description="Fingerprint detector with fingerprint database for comparison", 
        examples=[
            [os.path.abspath("examples/1.png"), 0.9],  
            [os.path.abspath("examples/2.png"), 0.9], 
        ],
        allow_flagging="never"
    )

    iface.launch(share=True)  



if __name__ == "__main__":
    # dtr = FingerprintDetector()
    # fingerprint_path = "examples/template_fingerprint.png"
    # dtr.detect(fingerprint_path)
    infer()