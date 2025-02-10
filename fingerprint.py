import cv2
import numpy as np

"""Список работ:
1) Интерфейс - gradio
2) Предобработка фото
3) Проверка размеров входных картинок
4) Извлечение фич (муниции) - почитать про отпечатки пальцев
5) Хранение отпечатков
    а. храним картинки в директории
    б. храним в виде вектор????
6) Методы сравнения отпечатков: хэш-суммой, расстояние Хэмминга, по узору, по особым точкам
7) Написать класс для всей программы

** Распознанование в real-time 
https://habr.com/ru/articles/116603/
https://github.com/kjanko/python-fingerprint-recognition/
https://www.codespeedy.com/fingerprint-detection-in-python/
https://www.kaggle.com/code/kairess/fingerprint-recognition
"""




# 1.  Функции обработки изображений:

def preprocess_image(image_path):
    """
    Предварительная обработка изображения отпечатка пальца:
    - Преобразование в оттенки серого
    - Бинаризация (преобразование в черно-белое изображение)
    - Улучшение контраста
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None

    # Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Бинаризация (Otsu's thresholding)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Инвертируем цвета (отпечаток должен быть черным на белом фоне)
    img = cv2.bitwise_not(img)

    return img


def extract_features(img):
    """
    Извлечение ключевых особенностей (минуций) из отпечатка пальца.
    В этом упрощенном примере, просто возвращает изображение как "вектор признаков".
    В реальных системах здесь используются более сложные алгоритмы (например, анализ хребтов, окончаний и разветвлений).
    """
    # TODO:  Заменить на более сложный алгоритм извлечения минуций.
    # Например, можно попробовать скелетизацию (thinning) и затем анализ точек ветвления и окончаний.
    return img


# 2.  Функция сравнения отпечатков:

def match_fingerprints(template_features, input_features, threshold=0.9):
    """
    Сравнение двух отпечатков пальцев на основе извлеченных признаков.
    В этом примере, просто сравнивает изображения с использованием корреляции.
    В реальных системах используется более сложный анализ соответствия минуций.
    """
    # Преобразуем в float32 для корректной работы cv2.matchTemplate
    template_features = template_features.astype(np.float32)
    input_features = input_features.astype(np.float32)

    # Сравнение с помощью корреляции
    result = cv2.matchTemplate(input_features, template_features, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)  # Получаем максимальное значение корреляции

    print(f"Значение корреляции: {max_val}")

    return max_val >= threshold


# 3.  Основная программа:

def main():
    """
    Основная функция программы.
    """

    # 1.  Загрузка образца отпечатка пальца (шаблона)
    template_path = "input_fingerprint.png"  # Замените на путь к вашему образцу
    template_img = preprocess_image(template_path)
    cv2.imshow("tesa",template_img)
    cv2.waitKey(0)


    if template_img is None:
        return
    template_features = extract_features(template_img)

    # 2.  Загрузка входного отпечатка пальца (для сравнения)
    input_path = "template_fingerprint.png"  # Замените на путь к отпечатку для сравнения
    input_img = preprocess_image(input_path)
    if input_img is None:
        return
    input_features = extract_features(input_img)

    # 3.  Сравнение отпечатков пальцев
    if match_fingerprints(template_features, input_features):
        print("Отпечатки пальцев совпадают.")
    else:
        print("Отпечатки пальцев не совпадают.")


if __name__ == "__main__":
    main()