# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import filtros


class AppFiltros:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Imagenes")
        self.root.geometry("1100x600")
        self.root.configure(bg="#1e1e1e")

        self.img_original = None
        self.img_resultado = None

        # -------- PANEL IZQUIERDO --------
        self.panel = tk.Frame(root, bg="#2b2b2b", width=200)
        self.panel.pack(side="left", fill="y")

        # Boton cargar
        tk.Button(self.panel, text="Cargar Imagen", command=self.cargar_imagen, bg="#4CAF50", fg="white").pack(pady=10, fill="x")

        # Botones filtros
        tk.Button(self.panel, text="Media", command=self.aplicar_media).pack(fill="x")
        tk.Button(self.panel, text="Mediana", command=self.aplicar_mediana).pack(fill="x")
        tk.Button(self.panel, text="Logaritmico", command=self.aplicar_log).pack(fill="x")
        tk.Button(self.panel, text="Normalizado", command=self.aplicar_normalizado).pack(fill="x")
        tk.Button(self.panel, text="Gaussiano", command=self.aplicar_gaussiano).pack(fill="x")
        tk.Button(self.panel, text="Laplace", command=self.aplicar_laplace).pack(fill="x")
        tk.Button(self.panel, text="Sobel", command=self.aplicar_sobel).pack(fill="x")
        tk.Button(self.panel, text="Canny", command=self.aplicar_canny).pack(fill="x")

        # -------- SLIDERS --------
        tk.Label(self.panel, text="Canny Min", bg="#2b2b2b", fg="white").pack()
        self.slider_canny_min = tk.Scale(self.panel, from_=0, to=255, orient="horizontal")
        self.slider_canny_min.set(100)
        self.slider_canny_min.pack(fill="x")

        tk.Label(self.panel, text="Canny Max", bg="#2b2b2b", fg="white").pack()
        self.slider_canny_max = tk.Scale(self.panel, from_=0, to=255, orient="horizontal")
        self.slider_canny_max.set(200)
        self.slider_canny_max.pack(fill="x")

        tk.Label(self.panel, text="Kernel Gauss", bg="#2b2b2b", fg="white").pack()
        self.slider_gauss = tk.Scale(self.panel, from_=1, to=15, orient="horizontal")
        self.slider_gauss.set(5)
        self.slider_gauss.pack(fill="x")

        # Boton guardar
        tk.Button(self.panel, text="Guardar Imagen", command=self.guardar_imagen, bg="#2196F3", fg="white").pack(pady=10, fill="x")

        # -------- PANEL IMAGENES --------
        self.frame_imgs = tk.Frame(root, bg="#1e1e1e")
        self.frame_imgs.pack(side="right", expand=True, fill="both")

        self.label_original = tk.Label(self.frame_imgs, bg="#1e1e1e")
        self.label_original.pack(side="left", padx=10)

        self.label_resultado = tk.Label(self.frame_imgs, bg="#1e1e1e")
        self.label_resultado.pack(side="right", padx=10)

    # -------- CARGAR --------
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imagenes", "*.jpg *.png *.jpeg *.bmp")]
        )

        if ruta:
            self.img_original = self.leer_imagen_segura(ruta)

            if self.img_original is None:
                print("Error cargando imagen")
                return

            self.img_resultado = self.img_original.copy()
            self.mostrar_imagenes()

    # -------- MOSTRAR --------
    def mostrar_imagenes(self):
        self.mostrar_en_label(self.img_original, self.label_original)
        self.mostrar_en_label(self.img_resultado, self.label_resultado)

    def mostrar_en_label(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((400, 400))

        img_tk = ImageTk.PhotoImage(img_pil)

        label.configure(image=img_tk)
        label.image = img_tk

    # -------- LECTURA SEGURA --------
    def leer_imagen_segura(self, ruta):
        data = np.fromfile(ruta, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # -------- FILTROS --------
    def aplicar_media(self):
        self.img_resultado = filtros.filtro_media(self.img_original)
        self.mostrar_imagenes()

    def aplicar_mediana(self):
        self.img_resultado = filtros.filtro_mediana(self.img_original)
        self.mostrar_imagenes()

    def aplicar_log(self):
        self.img_resultado = filtros.filtro_logaritmico(self.img_original)
        self.mostrar_imagenes()

    def aplicar_normalizado(self):
        self.img_resultado = filtros.filtro_normalizado(self.img_original)
        self.mostrar_imagenes()

    def aplicar_gaussiano(self):
        k = self.slider_gauss.get()
        if k % 2 == 0:
            k += 1
        self.img_resultado = cv2.GaussianBlur(self.img_original, (k, k), 0)
        self.mostrar_imagenes()

    def aplicar_laplace(self):
        self.img_resultado = filtros.filtro_laplace(self.img_original)
        self.mostrar_imagenes()

    def aplicar_sobel(self):
        self.img_resultado = filtros.filtro_sobel(self.img_original)
        self.mostrar_imagenes()

    def aplicar_canny(self):
        min_val = self.slider_canny_min.get()
        max_val = self.slider_canny_max.get()
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, min_val, max_val)
        self.img_resultado = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        self.mostrar_imagenes()

    # -------- GUARDAR --------
    def guardar_imagen(self):
        if self.img_resultado is None:
            return

        ruta = filedialog.asksaveasfilename(defaultextension=".png")
        if ruta:
            cv2.imwrite(ruta, self.img_resultado)


if __name__ == "__main__":
    root = tk.Tk()
    app = AppFiltros(root)
    root.mainloop()
