import cv2
import numpy as np
import os
import tqdm
import pickle
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


    def __init__(self, templates_path:str="base", load_base_file:bool=True, save_base_file:bool=True) -> None:
        """
        Описание:
        -
        - Заполняет базу отпечатков пальец, извлекает особенности.
        
        Параметры:
        -
        - templates_path: str: Путь до директории с фото;
        - load_base_file: bool: Флаг, включающий подгрузку базы из файла.
        - save_base_file: bool: Флаг, включающий сохранение базы в файл.

        Возвращает:
        -
        - None. 
        """

        self.__base_images = []
        self.__base_features = []
        self.__sift = cv2.SIFT_create(nfeatures=500)

        if load_base_file:
            print("Загрузка базы фото.")
            try:
                self.__base_images, self.__base_features = self.__load_data()
            except FileNotFoundError:
                print(f"Файл базы данных не найден: {self.base_file_path}. Инициализация пустой базы.")
                self.__base_images = []
                self.__base_features = []
            except Exception as e:
                print(f"Ошибка при загрузке базы данных: {e}.  Инициализация пустой базы.")
                self.__base_images = []
                self.__base_features = []
        else:
            print("Предобработка базы фото.")
            try:
                for filename in tqdm.tqdm(os.listdir(templates_path)):
                    filepath = os.path.join(templates_path, filename)
                    if os.path.isfile(filepath):
                        # Проверяем, является ли файл изображением (по расширению)
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')): 
                            self.__base_images.append((self.__preprocess_image(filepath), filename))

            except FileNotFoundError:
                print(f"Директория не найдена: {templates_path}")
            except OSError as e:
                print(f"Ошибка при доступе к директории {templates_path}: {e}")

            self.__extract_features_SIFT()

            if save_base_file:
                print("Сохранение базы фото.")
                self.__save_data(self.__base_images, self.__base_features)




    def __convert_keypoints_to_list(self, keypoints:cv2.KeyPoint) ->list[dict]:
        """
        Описание:
        -
        - Преобразует список cv2.KeyPoint в список словарей (или списков).
        
        Параметры:
        -
        - keypoints: cv2.KeyPoint: Список cv2.KeyPoint; 

        Возвращает:
        -
        - cv2.KeyPoint. 
        """
        keypoints_list = []
        for kp in keypoints:
            keypoints_list.append(dict(pt=kp.pt, size=kp.size, angle=kp.angle,
                                    response=kp.response, octave=kp.octave,
                                    class_id=kp.class_id))
        return keypoints_list





    def __convert_list_to_keypoints(self, keypoints_list:list) -> cv2.KeyPoint:
        """
        Описание:
        -
        - Преобразует список словарей обратно в список cv2.KeyPoint.
        
        Параметры:
        -
        - keypoints_list: list: Список с ключевыми точками; 

        Возвращает:
        -
        - cv2.KeyPoint. 
        """
        keypoints = []
        for kp_data in keypoints_list:
            kp = cv2.KeyPoint(
                x=float(kp_data['pt'][0]),
                y=float(kp_data['pt'][1]),
                size=float(kp_data['size']),  
                angle=float(kp_data['angle']),
                response=float(kp_data['response']),
                octave=int(kp_data['octave']),
                class_id=int(kp_data['class_id'])
            )
            keypoints.append(kp)
        return keypoints





    def __save_data(self, data:tuple, base_features:tuple, filename:str="Fingerprint/base.pkl") -> None:
        """
        Описание:
        -
        - Сохранение базы в файл.
        
        Параметры:
        -
        - data: tuple: Кортеж с фото; 
        - base_features: Кортеж с фичами отпечатков;
        - filename: str: Путь файла для сохранения.

        Возвращает:
        -
        - None. 
        """
        # Преобразуем keypoints в списки перед сохранением
        base_features_serializable = []
        for keypoints, descriptors in base_features:
            keypoints_list = self.__convert_keypoints_to_list(keypoints)
            base_features_serializable.append((keypoints_list, descriptors))

        # Сохраняем данные
        try:
            with open(filename, 'wb') as f:
                pickle.dump((data, base_features_serializable), f)
            print(f"Данные сохранены в {filename} (pickle)")
        except Exception as e:
            print(f"Ошибка при сохранении данных в {filename}: {e}")





    # Функция для загрузки данных из файла
    def __load_data(self, filename:str="Fingerprint/base.pkl") -> tuple[list, list]:
        """
        Описание:
        -
        - Загрузка базы отпечатков из файла.
        
        Параметры:
        -
        - filename: str: Путь до файла. 

        Возвращает:
        -
        - None. 
        """
        try:
            with open(filename, 'rb') as f:
                data, base_features_serializable = pickle.load(f)

            # Преобразуем списки обратно в cv2.KeyPoint
            base_features = []
            for keypoints_list, descriptors in base_features_serializable:
                keypoints = self.__convert_list_to_keypoints(keypoints_list)
                base_features.append((keypoints, descriptors))

            print(f"Данные загружены из {filename} (pickle)")
            return data, base_features
        except FileNotFoundError:
            raise  
        except Exception as e:
            raise  





    def __preprocess_image(self, image_path:str) -> np.ndarray:
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

        # Уменьшаем размер изображения
        # img = cv2.resize(img, (320, 240))  # Измените размер по необходимости
        # # Инвертируем цвета (отпечаток должен быть черным на белом фоне)
        # img = cv2.bitwise_not(img)

        return img
    



    def __extract_features_SIFT(self):
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
        self.__base_features = []  # Clear previous features
        for image_data in tqdm.tqdm(self.__base_images):
            image = image_data[0]
            try:
                keypoints_2, descriptors_2 = self.__sift.detectAndCompute(image, None)
                if keypoints_2 is not None and descriptors_2 is not None:
                    self.__base_features.append((keypoints_2, descriptors_2))
                else:
                    print(f"Не удалось найти ключевые точки для изображения {image_data[1]}")
                    self.__base_features.append(([], None)) 

            except Exception as e:
                print(f"Ошибка при вычислении SIFT для изображения {image_data[1]}: {e}")
                self.__base_features.append(([], None))  


    

    def SIFT_match(self, image:np.ndarray, threshold:float=0.90) -> tuple | None:
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

        keypoints_1, descriptors_1 = self.__sift.detectAndCompute(img, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)


        for i, tt in enumerate(self.__base_features):
            
            try:
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
                    result = cv2.drawMatches(img, keypoints_1, self.__base_images[i][0], tt[0], match_points, None) 
                    result = cv2.resize(result, None, fx=2.5, fy=2.5)
                    return result, img, self.__base_images[i][0], "".join(f"Match Figerprint ID:{i}; {self.__base_images[i][1]} - {len(match_points) / keypoints * 100}%")
                
            except Exception as e:
                print(f"Ошибка при сопоставлении признаков для {i}: {e}")
                
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
        # Здесь нужен цикл которые пробегает по self.__base_images и методом ниже производит сравнение
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

