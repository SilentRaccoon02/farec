import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
import threading


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система распознавания лиц - DCT")
        self.root.geometry("1200x800")

        self.reference_features = {}
        self.reference_sample_paths = {}
        self.dct_mat = None
        self.processing = False

        self.create_widgets()

    def create_widgets(self):
        # Фрейм параметров
        params_frame = ttk.LabelFrame(self.root, text="Параметры алгоритма", padding=10)
        params_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(params_frame, text="Порог:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.threshold_var = tk.DoubleVar(value=0.7)
        ttk.Entry(params_frame, textvariable=self.threshold_var, width=10).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(params_frame, text="Размер блока:").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        self.block_size_var = tk.IntVar(value=8)
        ttk.Entry(params_frame, textvariable=self.block_size_var, width=10).grid(
            row=0, column=3, padx=5, pady=5
        )

        ttk.Label(params_frame, text="Число коэффициентов:").grid(
            row=0, column=4, padx=5, pady=5, sticky="w"
        )
        self.num_coeffs_var = tk.IntVar(value=24)
        ttk.Entry(params_frame, textvariable=self.num_coeffs_var, width=10).grid(
            row=0, column=5, padx=5, pady=5
        )

        # Фрейм обучения
        train_frame = ttk.LabelFrame(self.root, text="Обучение", padding=10)
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            train_frame,
            text="Выбрать директорию с эталонами",
            command=self.select_reference_directory,
        ).pack(side="left", padx=5)
        self.ref_dir_label = ttk.Label(train_frame, text="Не выбрана")
        self.ref_dir_label.pack(side="left", padx=5)

        ttk.Button(
            train_frame, text="Старт обучения", command=self.start_training
        ).pack(side="left", padx=5)

        self.train_status_label = ttk.Label(train_frame, text="")
        self.train_status_label.pack(side="left", padx=5)

        # Фрейм распознавания
        recog_frame = ttk.LabelFrame(self.root, text="Распознавание", padding=10)
        recog_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            recog_frame,
            text="Загрузить изображение для распознавания",
            command=self.recognize_face,
        ).pack(side="left", padx=5)
        self.recog_status_label = ttk.Label(recog_frame, text="")
        self.recog_status_label.pack(side="left", padx=5)

        # Фрейм тестирования
        test_frame = ttk.LabelFrame(self.root, text="Тестирование", padding=10)
        test_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(
            test_frame,
            text="Выбрать директорию для тестирования",
            command=self.test_system,
        ).pack(side="left", padx=5)
        self.test_status_label = ttk.Label(test_frame, text="")
        self.test_status_label.pack(side="left", padx=5)

        # Фрейм результатов
        results_frame = ttk.LabelFrame(
            self.root, text="Результаты сравнения", padding=10
        )
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Контейнер для изображений
        images_container = ttk.Frame(results_frame)
        images_container.pack(fill="both", expand=True)

        # Тестовое изображение
        test_img_frame = ttk.Frame(images_container)
        test_img_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(test_img_frame, text="Тестовое изображение").pack()
        self.test_image_label = ttk.Label(test_img_frame)
        self.test_image_label.pack(fill="both", expand=True)

        # Эталонное изображение
        ref_img_frame = ttk.Frame(images_container)
        ref_img_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(ref_img_frame, text="Наиболее похожий эталон").pack()
        self.ref_image_label = ttk.Label(ref_img_frame)
        self.ref_image_label.pack(fill="both", expand=True)

        # Текст результата
        self.result_text = tk.Text(results_frame, height=5, wrap="word")
        self.result_text.pack(fill="x", padx=5, pady=5)

    def create_dct_matrix(self, N):
        dct_mat = np.zeros((N, N))
        for k in range(N):
            alpha = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            for n in range(N):
                dct_mat[k, n] = alpha * np.cos((2 * n + 1) * k * np.pi / (2 * N))
        return dct_mat

    def dct_2d_fast(self, block, dct_mat):
        return dct_mat @ block @ dct_mat.T

    def extract_dct_features(self, image_path, dct_mat, block_size, num_coefficients):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        img = cv2.resize(img, (128, 128))
        img = cv2.equalizeHist(img)
        img = img.astype(float)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        features = []
        h, w = img.shape
        zigzag_size = int(np.ceil(np.sqrt(num_coefficients * 2)))

        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i : i + block_size, j : j + block_size]
                dct_block = self.dct_2d_fast(block, dct_mat)

                coeffs = []
                for k in range(min(zigzag_size, block_size)):
                    for l in range(min(zigzag_size - k, block_size)):
                        if len(coeffs) < num_coefficients:
                            coeffs.append(dct_block[k, l])

                features.extend(coeffs)

        return np.array(features)

    def similarity_metric(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)

        euclidean_dist = np.linalg.norm(vec1 - vec2)
        max_dist = np.sqrt(len(vec1)) * 2
        euclidean_sim = 1 - (euclidean_dist / max_dist)

        return cosine_sim * 0.7 + euclidean_sim * 0.3

    def select_reference_directory(self):
        directory = filedialog.askdirectory(title="Выберите директорию с эталонами")
        if directory:
            self.reference_directory = directory
            self.ref_dir_label.config(text=os.path.basename(directory))

    def start_training(self):
        if not hasattr(self, "reference_directory"):
            messagebox.showerror("Ошибка", "Выберите директорию с эталонами")
            return

        if self.processing:
            messagebox.showwarning("Предупреждение", "Обработка уже выполняется")
            return

        def train():
            self.processing = True
            self.train_status_label.config(text="Обучение...")

            try:
                block_size = self.block_size_var.get()
                num_coefficients = self.num_coeffs_var.get()

                self.dct_mat = self.create_dct_matrix(block_size)
                self.reference_features = {}
                self.reference_sample_paths = {}

                person_dirs = [
                    d for d in Path(self.reference_directory).iterdir() if d.is_dir()
                ]

                for person_dir in person_dirs:
                    person_name = person_dir.name
                    image_files = (
                        list(person_dir.glob("*.jpg"))
                        + list(person_dir.glob("*.png"))
                        + list(person_dir.glob("*.jpeg"))
                    )

                    if not image_files:
                        continue

                    person_features = []
                    for img_path in image_files:
                        try:
                            features = self.extract_dct_features(
                                str(img_path),
                                self.dct_mat,
                                block_size,
                                num_coefficients,
                            )
                            person_features.append(features)
                        except Exception as e:
                            print(f"Ошибка обработки {img_path}: {e}")

                    if person_features:
                        self.reference_features[person_name] = person_features
                        self.reference_sample_paths[person_name] = str(image_files[0])

                self.train_status_label.config(
                    text=f"Обучение завершено. Загружено {len(self.reference_features)} человек"
                )
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
                self.train_status_label.config(text="Ошибка обучения")
            finally:
                self.processing = False

        threading.Thread(target=train, daemon=True).start()

    def recognize_face(self):
        if not self.reference_features:
            messagebox.showerror("Ошибка", "Сначала выполните обучение")
            return

        image_path = filedialog.askopenfilename(
            title="Выберите изображение", filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not image_path:
            return

        try:
            block_size = self.block_size_var.get()
            num_coefficients = self.num_coeffs_var.get()
            threshold = self.threshold_var.get()

            test_features = self.extract_dct_features(
                image_path, self.dct_mat, block_size, num_coefficients
            )

            max_similarity = 0.0
            best_match = None

            for person_name, features_list in self.reference_features.items():
                for ref_features in features_list:
                    min_len = min(len(test_features), len(ref_features))
                    similarity = self.similarity_metric(
                        test_features[:min_len], ref_features[:min_len]
                    )

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = person_name

            # Отображение результатов
            self.display_comparison(image_path, best_match, max_similarity, threshold)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания: {str(e)}")

    def display_comparison(self, test_path, best_match, similarity, threshold):
        # Загрузка и отображение тестового изображения
        test_img = Image.open(test_path)
        test_img.thumbnail((400, 400))
        test_photo = ImageTk.PhotoImage(test_img)
        self.test_image_label.config(image=test_photo)
        self.test_image_label.image = test_photo

        # Загрузка и отображение эталонного изображения
        if best_match and best_match in self.reference_sample_paths:
            ref_img = Image.open(self.reference_sample_paths[best_match])
            ref_img.thumbnail((400, 400))
            ref_photo = ImageTk.PhotoImage(ref_img)
            self.ref_image_label.config(image=ref_photo)
            self.ref_image_label.image = ref_photo

        # Текстовый результат
        is_match = similarity >= threshold
        result_text = f"Результат: {'Совпадение' if is_match else 'Не совпадает'}\n"
        result_text += (
            f"Наиболее похож на: {best_match if best_match else 'Неизвестно'}\n"
        )
        result_text += f"Сходство: {similarity:.4f}\n"
        result_text += f"Порог: {threshold:.4f}"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)

    def test_system(self):
        if not self.reference_features:
            messagebox.showerror("Ошибка", "Сначала выполните обучение")
            return

        test_directory = filedialog.askdirectory(
            title="Выберите директорию для тестирования"
        )
        if not test_directory:
            return

        def test():
            self.test_status_label.config(text="Тестирование...")

            try:
                block_size = self.block_size_var.get()
                num_coefficients = self.num_coeffs_var.get()
                threshold = self.threshold_var.get()

                test_files = (
                    list(Path(test_directory).glob("*.jpg"))
                    + list(Path(test_directory).glob("*.png"))
                    + list(Path(test_directory).glob("*.jpeg"))
                )

                correct = 0
                total = 0

                for test_file in test_files:
                    true_label = test_file.stem

                    try:
                        test_features = self.extract_dct_features(
                            str(test_file), self.dct_mat, block_size, num_coefficients
                        )

                        max_similarity = 0.0
                        best_match = None

                        for (
                            person_name,
                            features_list,
                        ) in self.reference_features.items():
                            for ref_features in features_list:
                                min_len = min(len(test_features), len(ref_features))
                                similarity = self.similarity_metric(
                                    test_features[:min_len], ref_features[:min_len]
                                )

                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_match = person_name

                        if best_match == true_label and max_similarity >= threshold:
                            correct += 1

                        total += 1
                    except Exception as e:
                        print(f"Ошибка обработки {test_file}: {e}")

                accuracy = (correct / total * 100) if total > 0 else 0

                self.test_status_label.config(
                    text=f"Точность: {accuracy:.2f}% ({correct}/{total})"
                )
                messagebox.showinfo(
                    "Результаты тестирования",
                    f"Точность: {accuracy:.2f}%\nПравильно распознано: {correct} из {total}",
                )

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка тестирования: {str(e)}")
                self.test_status_label.config(text="Ошибка тестирования")

        threading.Thread(target=test, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
