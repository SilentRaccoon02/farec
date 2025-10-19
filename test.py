import os
import glob
from collections import defaultdict
from typing import List, Tuple
import random

# ============================================
# НАСТРОЙКИ - задайте все параметры здесь
# ============================================
DIRECTORY = "path/to/your/images"  # путь к директории с изображениями
NUM_REFERENCE = 8  # количество эталонных фото
NUM_TEST = 2  # количество тестовых фото
BLOCK_SIZE = 8  # размер блока для DCT
NUM_COEFFICIENTS = 15  # количество коэффициентов
THRESHOLD = 0.85  # порог распознавания
RANDOM_SEED = 42  # seed для воспроизводимости результатов (None для случайности)
# ============================================


def test_face_recognition(directory: str, face_recognition_func):
    """
    Тестирование функции распознавания лиц

    Args:
        directory: путь к директории с изображениями
        face_recognition_func: функция распознавания лиц
    """

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Сканирование директории и группировка по ID человека
    image_files = glob.glob(os.path.join(directory, "*_*.jpg"))
    people_images = defaultdict(list)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Извлекаем ID человека (цифра после _)
        person_id = filename.split("_")[1].split(".")[0]
        people_images[person_id].append(img_path)

    print(f"Найдено {len(people_images)} человек")
    print(
        f"Параметры: эталоны={NUM_REFERENCE}, тесты={NUM_TEST}, block_size={BLOCK_SIZE}, coefficients={NUM_COEFFICIENTS}, threshold={THRESHOLD}\n"
    )

    # Тест 1: распознавание своих фото
    print("=" * 60)
    print("ТЕСТ 1: Распознавание фото того же человека")
    print("=" * 60)

    correct_recognitions = 0
    total_tests = 0

    for person_id, images in people_images.items():
        required_photos = NUM_REFERENCE + NUM_TEST
        if len(images) < required_photos:
            print(
                f"ОШИБКА: Недостаточно фото для человека {person_id} (требуется {required_photos}, есть {len(images)})"
            )
            continue

        # Случайное разделение на эталоны и тестовые
        shuffled = images.copy()
        random.shuffle(shuffled)
        reference_images = shuffled[:NUM_REFERENCE]
        test_images = shuffled[NUM_REFERENCE : NUM_REFERENCE + NUM_TEST]

        for test_img in test_images:
            result, similarity = face_recognition_func(
                test_img,
                reference_images,
                block_size=BLOCK_SIZE,
                num_coefficients=NUM_COEFFICIENTS,
                threshold=THRESHOLD,
            )

            if result:
                correct_recognitions += 1
            else:
                # Логируем только ошибки
                print(
                    f"✗ Человек {person_id}: {os.path.basename(test_img)} -> НЕ распознан (сходство: {similarity:.4f})"
                )

            total_tests += 1

    accuracy_1 = (correct_recognitions / total_tests * 100) if total_tests > 0 else 0
    print(
        f"\nРезультат теста 1: {correct_recognitions}/{total_tests} ({accuracy_1:.1f}%)\n"
    )

    # Тест 2: отклонение чужих фото
    print("=" * 60)
    print("ТЕСТ 2: Отклонение фото других людей")
    print("=" * 60)

    correct_rejections = 0
    total_tests_2 = 0

    person_ids = list(people_images.keys())

    for person_id in person_ids:
        images = people_images[person_id]
        if len(images) < NUM_REFERENCE:
            continue

        # эталонных фото текущего человека
        shuffled = images.copy()
        random.shuffle(shuffled)
        reference_images = shuffled[:NUM_REFERENCE]

        # Тестируем по одному фото каждого другого человека
        for other_person_id in person_ids:
            if other_person_id == person_id:
                continue

            other_images = people_images[other_person_id]
            if len(other_images) == 0:
                continue

            # Берем случайное фото другого человека
            test_img = random.choice(other_images)

            result, similarity = face_recognition_func(
                test_img,
                reference_images,
                block_size=BLOCK_SIZE,
                num_coefficients=NUM_COEFFICIENTS,
                threshold=THRESHOLD,
            )

            if not result:  # Правильно отклонено
                correct_rejections += 1
            else:
                # Логируем только ошибки
                print(
                    f"✗ Эталон: человек {person_id}, Тест: человек {other_person_id} -> ОШИБОЧНО распознан (сходство: {similarity:.4f})"
                )

            total_tests_2 += 1

    accuracy_2 = (correct_rejections / total_tests_2 * 100) if total_tests_2 > 0 else 0
    print(
        f"\nРезультат теста 2: {correct_rejections}/{total_tests_2} ({accuracy_2:.1f}%)\n"
    )

    # Итоговая статистика
    print("=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Тест 1 (распознавание своих): {accuracy_1:.1f}%")
    print(f"Тест 2 (отклонение чужих): {accuracy_2:.1f}%")
    total_correct = correct_recognitions + correct_rejections
    total_all = total_tests + total_tests_2
    overall_accuracy = (total_correct / total_all * 100) if total_all > 0 else 0
    print(f"Общая точность: {total_correct}/{total_all} ({overall_accuracy:.1f}%)")


# Использование:
if __name__ == "__main__":
    from fun import face_recognition_dct  # замените на реальный импорт

    test_face_recognition(DIRECTORY, face_recognition_dct)
