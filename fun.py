import numpy as np
import cv2


def face_recognition_dct(
    test_image_path,
    reference_images_paths,
    block_size=16,
    num_coefficients=25,
    threshold=0.92,
):
    """
    Распознавание лиц с использованием DCT (оптимизированная версия).

    Параметры:
    - test_image_path: путь к тестовому изображению
    - reference_images_paths: список путей к эталонным изображениям
    - block_size: размер блока для DCT (по умолчанию 16x16)
    - num_coefficients: количество коэффициентов DCT для сравнения
    - threshold: порог схожести (от 0 до 1)

    Возвращает:
    - (True/False, max_similarity): совпадение и максимальное сходство
    """

    def create_dct_matrix(N):
        """Предвычисление DCT матрицы для ускорения"""
        dct_mat = np.zeros((N, N))
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            for n in range(N):
                dct_mat[k, n] = alpha * np.cos((2 * n + 1) * k * np.pi / (2 * N))
        return dct_mat

    def dct_2d_fast(block, dct_mat):
        """Быстрое вычисление 2D DCT через матричное умножение"""
        return dct_mat @ block @ dct_mat.T

    def extract_dct_features(image_path, dct_mat):
        # Загрузка и предобработка изображения
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Изменение размера для унификации
        img = cv2.resize(img, (128, 128))

        # Выравнивание гистограммы и нормализация
        img = cv2.equalizeHist(img)
        img = img.astype(float)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        # Разбиение на блоки и применение DCT
        features = []
        h, w = img.shape

        # Вычисление количества коэффициентов для зигзаг-извлечения
        zigzag_size = int(np.ceil(np.sqrt(num_coefficients * 2)))

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i : i + block_size, j : j + block_size]

                # Применение 2D DCT (быстрая версия)
                dct_block = dct_2d_fast(block, dct_mat)

                # Извлечение коэффициентов в зигзаг-порядке (упрощенно)
                coeffs = []
                for k in range(min(zigzag_size, block_size)):
                    for l in range(min(zigzag_size - k, block_size)):
                        if len(coeffs) < num_coefficients:
                            coeffs.append(dct_block[k, l])

                features.extend(coeffs)

        return np.array(features)

    def similarity_metric(vec1, vec2):
        """Комбинированная метрика: евклидово расстояние + косинусное сходство"""
        # Косинусное сходство
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)

        # Нормализованное евклидово расстояние
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        max_dist = np.sqrt(len(vec1)) * 2  # Примерный максимум
        euclidean_sim = 1 - (euclidean_dist / max_dist)

        # Среднее двух метрик
        return cosine_sim * 0.7 + euclidean_sim * 0.3

    # Предвычисление DCT матрицы
    dct_mat = create_dct_matrix(block_size)

    # Извлечение признаков тестового изображения
    test_features = extract_dct_features(test_image_path, dct_mat)

    # Сравнение с эталонами
    max_similarity = 0.0

    for ref_path in reference_images_paths:
        ref_features = extract_dct_features(ref_path, dct_mat)

        # Приведение к одинаковой длине
        min_len = min(len(test_features), len(ref_features))
        similarity = similarity_metric(test_features[:min_len], ref_features[:min_len])

        max_similarity = max(max_similarity, similarity)

    is_match = max_similarity >= threshold

    return is_match, max_similarity


# Пример использования:
# is_match, similarity = face_recognition_dct(
#     'test_face.jpg',
#     ['reference1.jpg', 'reference2.jpg'],
#     block_size=16,
#     num_coefficients=25,
#     threshold=0.92
# )
# print(f"Результат: {'Тот же человек' if is_match else 'Другой человек'}")
# print(f"Максимальное сходство: {similarity:.4f}")
