from Fingerprint.fingerprint import FingerprintDetector
import gradio as gr  
import os



def infer() -> None:
    """
    Запускает веб-приложение

    Параметры: None

    Возвращает: None
    """

    #TODO
    #Продумай GUI.
    #Нужен выпадающий список позволяющий выбрать метод SIFT или Корреляцию
    #По сути условие: если метод этот то такой outputs[], другой значит и outputs другой
    #Также нужен слайдер для threshold (порога)


    dtr = FingerprintDetector()
    # Определяем веб-интерфейс Gradio
    iface = gr.Interface(
        fn=dtr.SIFT_match,  # Указываем функцию , которая будет вызываться при загрузке изображения.
        inputs=[
            gr.Image(type="numpy", label="Загруженное изображение"),  # Задаем тип входных данных.
        ],
        
        outputs=[
            gr.Image(type="numpy", label="Результат"),  # Первое выходное значение - это изображение
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

    iface.launch(share=True)  



if __name__ == "__main__":

    infer()