from Fingerprint.fingerprint import FingerprintDetector
import gradio as gr  
import os



def infer() -> None:
    """
    Запускает веб-приложение

    Параметры: None

    Возвращает: None
    """


    dtr = FingerprintDetector()
    # Определяем веб-интерфейс Gradio

    iface = gr.Interface(
        fn=dtr.inference,  # Указываем функцию , которая будет вызываться при загрузке изображения.
        inputs=[
            gr.Image(type="numpy", label="Загруженное изображение"),  # Задаем тип входных данных.
            gr.Slider(minimum=0, maximum=1, value=0.90, label="Порог уверенности совпадения"),  # Добавляем слайдер для регулировки порога уверенности.
            gr.Dropdown(["Correlation_match", "SIFT_match"], label="Выберите метод сравнения", value="SIFT_match"),
        ],
        
        outputs=[
            gr.Image(type="numpy", label="Результат"),
            gr.Image(type="numpy", label="Запрос"), 
            gr.Image(type="numpy", label="Совпадение"),
            gr.Textbox(label="Вывод")  # Второе выходное значение - список объектов
        ], 
        title="Fingerprint detector",  
        description="Fingerprint detector with fingerprint database for comparison", 
        examples=[
            [os.path.abspath("examples/1.png"), 0.9],  
            [os.path.abspath("examples/2.png"), 0.9],
            [os.path.abspath("examples/3.png"), 0.9], 
        ],
        allow_flagging="never"
    )


    iface.launch(share=False)  



if __name__ == "__main__":

    infer()