import cv2
import numpy as np
import os
import tqdm
"""Список работ:
1) Интерфейс - gradio +
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


class FingerprintDetector:


    def __init__(self, templates_path:str="base") -> None:
        """
        Описание:
        -
        - Заполняет базу отпечатков пальец, извлекает особенности.
        
        Параметры:
        -
        - templates_path: str: Путь до директории с фото.

        Возвращает:
        -
        - None. 
        """

        self.base_images = []
        self.base_features = []
        self.sift = cv2.SIFT_create()
        print("Предобработка базы фото.")
        for filename in tqdm.tqdm(os.listdir(templates_path)):
            filepath = os.path.join(templates_path, filename)
            if os.path.isfile(filepath):
                # Проверяем, является ли файл изображением (по расширению)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')): 
                    self.base_images.append((self.preprocess_image(filepath), filename))
        
        self.extract_features()
        



    def preprocess_image(self, image_path:str) -> np.ndarray:
        """
        Описание:
        -
        - Предварительная обработка фото.
        
        Параметры:
        -
        - image_path: str: Путь до фото. 

        Возвращает:
        -
        - np.ndarray. 
        """

        #TODO
        # Причесать, улучшить, сделать так, чтобы изображение стало чётче и "выразительнее"

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return None

        # Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Бинаризация (Otsu's thresholding)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Инвертируем цвета (отпечаток должен быть черным на белом фоне)
        # img = cv2.bitwise_not(img)

        return img
    



    def extract_features(self):
        """
        Описание:
        -
        -Получение ключевых точек на изображении.
        
        Параметры:
        -
        - None. 

        Возвращает:
        -
        - None. 
        """
        print("Получение ключевых точек на фото из базы.")
        # Проходим по изображениями и находим ключевые точки
        for image in tqdm.tqdm(self.base_images): 
            keypoints_2, descriptors_2 = self.sift.detectAndCompute(image[0], None)
            self.base_features.append((keypoints_2, descriptors_2))


    

    def SIFT_match(self, image:np.ndarray, threshold:float=0.95) -> tuple | None:
        """
        Описание:
        -
        -Поиск совпадений методом SIFT.
        
        Параметры:
        -
        - image: np.ndarray : Входное изображение; 
        - threshold: float : Порог.

        Возвращает:
        -
        - tuple | None. 
        """

        img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Бинаризация (Otsu's thresholding)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        keypoints_1, descriptors_1 = self.sift.detectAndCompute(img, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)


        for i, tt in enumerate(self.base_features):

            matches = flann.knnMatch(descriptors_1, tt[1], k=2)
            match_points = []
        
            for p, q in matches:
                if p.distance < 0.1*q.distance:
                    match_points.append(p)


            keypoints = 0
            if len(keypoints_1) <= len(tt[1]):
                keypoints = len(keypoints_1)            
            else:
                keypoints = len(tt[1])
            if (len(match_points) / keypoints) >= threshold:
                result = cv2.drawMatches(img, keypoints_1, self.base_images[i][0], tt[0], match_points, None) 
                result = cv2.resize(result, None, fx=2.5, fy=2.5)
                return result, img, self.base_images[i][0], "".join(f"Match Figerprint ID:{i}; {self.base_images[i][1]} - {len(match_points) / keypoints * 100}%")
                
        return None, None, None, "Совпадений нет!"






    def match_fingerprints(self, image:np.ndarray, threshold:float=0.95) -> tuple | None:
        """
        Описание:
        -
        -Поиск совпадений корреляционным методом.
        
        Параметры:
        -
        -

        Возвращает:
        -
        - tuple | None. 
        """

        # TODO
        # Доделать логику
        # Здесь нужен цикл которые пробегает по self.base_images и методом ниже производит сравнение
        # Ну и отсечь надо порогом.
        # Изображение из базы которое совпало, изображение запрос, и текст по аналогии с методом выше
        # По дефолту retunr None, None, "Совпадений нет!"
        # Обрати внимание количество возвращаемых объектов здесь отличается от предыдущего метода, соответственно и другой конфиг Gradio





        # Преобразуем в float32 для корректной работы cv2.matchTemplate
        template_features = template_features.astype(np.float32)
        input_features = input_features.astype(np.float32)

        # Сравнение с помощью корреляции
        result = cv2.matchTemplate(input_features, template_features, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)  # Получаем максимальное значение корреляции

        print(f"Значение корреляции: {max_val}")

        return max_val >= threshold

