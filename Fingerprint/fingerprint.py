import cv2
import numpy as np
import os
import tqdm
import pickle


class FingerprintDetector:


    def __init__(self, templates_path:str="base", load_base_file:bool=True, save_base_file:bool=True, target_width:int=800, target_height:int=1200) -> None:
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
        self.__target_width = target_width
        self.__target_height = target_height

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





            
    def __should_invert(self, img:np.ndarray) -> bool:
        """
        Определяет, нужно ли инвертировать изображение, основываясь на его фоне.

        Параметры:
        - img: np.ndarray: Исходное изображение.

        Возвращает:
        - bool: True, если изображение должно быть инвертировано, иначе False.
        """
        h, w = img.shape
        corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
        mean_corners = np.mean(corners)
        
        if mean_corners > 200:
            return False
        
        mean_brightness = np.mean(img)
        
        if mean_brightness < 100:
            return True

        return False
    



    def __resize_with_padding(self, img:np.ndarray) -> np.ndarray:
        """
        Приводит изображение к заданному размеру с добавлением паддинга.

        Параметры:
        - img: np.ndarray: Исходное изображение.
        - target_size: tuple: Целевой размер изображения (ширина, высота).

        Возвращает:
        -
        """
        h, w = img.shape[:2]
        target_w, target_h = (self.__target_width, self.__target_height)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return img_padded
    




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

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return None
            
        # 1. Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(24, 24))
        img = clahe.apply(img)

        # 2. Бинаризация (Otsu's thresholding)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

         # 3. Проверка фона (если белый - инвертируем)
        if self.__should_invert(img):
            img = cv2.bitwise_not(img)

        # 4. Приведение к единому размеру
        #img = self.__resize_with_padding(img)

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

        # 1. Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(24, 24))
        img = clahe.apply(img)

        # 2. Бинаризация (Otsu's thresholding)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

         # 3. Проверка фона (если белый - инвертируем)
        if self.__should_invert(img):
            img = cv2.bitwise_not(img)

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






    def Correlation_match(self, image:np.ndarray, threshold:float=0.95) -> tuple | None:
        """
        Сравнивает заданное изображение с изображениями базы данных с использованием корреляции.
        Если совпадение превышает порог, возвращает изображение совпадения, исходное изображение и сообщение.

        Параметры:
        -
        image: np.ndarray
            - Изображение для сравнения.
        threshold: float, по умолчанию 0.95
            - Порог корреляции для определения совпадения.

        Возвращает:
        -
        tuple | None:
            - Кортеж с изображением совпадения, исходным изображением и сообщением о совпадении, если найдено.
            - `None, None, "Совпадений нет!"`, если совпадений нет.
        """
        best_score = 0
        best_match = None
        best_match_image = None

        img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Улучшение контраста (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(24, 24))
        img = clahe.apply(img)

        # 2. Бинаризация (Otsu's thresholding)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

         # 3. Проверка фона (если белый - инвертируем)
        if self.__should_invert(img):
            img = cv2.bitwise_not(img)

        #img = img.astype(np.float32)

        # Преобразуем в оттенки серого, если они цветные
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Преобразуем в CV_8U
        img = img.astype(np.uint8)
       


        for base_image, filename in self.__base_images:
            try:

                # Преобразуем в float32 для корректной работы cv2.matchTemplate
                base_image = base_image.astype(np.float32)
                if len(base_image.shape) > 2:
                    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

                base_image = base_image.astype(np.uint8)

                # Сравнение с помощью корреляции
                result = cv2.matchTemplate(img, base_image, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)  # Получаем максимальное значение корреляции

                print(f"Значение корреляции с {filename}: {max_val}")

                if max_val > best_score and max_val >= threshold:
                    best_score = max_val
                    best_match = filename
                    best_match_image = base_image
            
            except Exception as e:
                print(f"Ошибка при сравнении с {filename}: {e}")

        if best_match:
            return None, img, best_match_image, f"Совпадение найдено: {best_match} ({best_score * 100:.2f}%)"
        
        return None, None, None, "Совпадений нет!"
        




    def inference(self, image:np.ndarray, threshold:float=0.95, method:str="SIFT_match") -> None:

        if method == "SIFT_match":
            return self.SIFT_match(image, threshold)
        elif method == "Correlation_match":
            return self.Correlation_match(image, threshold)
