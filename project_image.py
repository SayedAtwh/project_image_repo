import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os


class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.root.geometry("1200x800")

        self.original_cv = None
        self.current_cv = None

        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self.root, bg="#f0f0f0", height=100)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Button(
            top_frame, text="📁 Browse Image",
            command=self.browse_image,
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top_frame, text="🔄 Reset Image",
            command=self.reset_image,
            bg="#FF9800", fg="white", font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top_frame, text="💾 Save Image",
            command=self.save_image,
            bg="#2196F3", fg="white", font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(top_frame, text="No image loaded", bg="#f0f0f0")
        self.status_label.pack(side=tk.LEFT, padx=20)

        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = tk.Frame(content_frame, bg="gray")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame, bg="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        filter_frame = tk.Frame(content_frame, bg="#f0f0f0", width=300)
        filter_frame.pack(side=tk.RIGHT, fill=tk.Y)
        filter_frame.pack_propagate(False)

        tk.Label(filter_frame, text="Filters & Effects",
                 bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)

        canvas = tk.Canvas(filter_frame, bg="#f0f0f0")
        scrollbar = tk.Scrollbar(filter_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.create_section(scrollable_frame, "NOISE", [
            ("Salt Noise", lambda: self.add_noise_dialog("salt")),
            ("Pepper Noise", lambda: self.add_noise_dialog("pepper")),
            ("Salt & Pepper", lambda: self.add_noise_dialog("both")),
        ])

        self.create_section(scrollable_frame, "FILTERS", [
            ("Min Filter", lambda: self.apply_filter_dialog("min")),
            ("Max Filter", lambda: self.apply_filter_dialog("max")),
            ("Median Filter", lambda: self.apply_filter_dialog("median")),
            ("Mean Filter", lambda: self.apply_filter_dialog("mean")),
        ])

        self.create_section(scrollable_frame, "EDGE DETECTION", [
            ("Laplacian", lambda: self.apply_edge_filter("laplacian")),
            ("Sobel", lambda: self.apply_edge_filter("sobel")),
            ("Prewitt", lambda: self.apply_edge_filter("prewitt")),
            ("Roberts", lambda: self.apply_edge_filter("roberts")),
        ])

        self.create_section(scrollable_frame, "ENHANCEMENT", [
            ("High Boost", lambda: self.apply_filter_dialog("highboost")),
            ("Unsharp Mask", lambda: self.apply_filter_dialog("unsharp")),
        ])

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_section(self, parent, title, buttons):
        frame = tk.LabelFrame(parent, text=title, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=5, pady=5)

        for text, cmd in buttons:
            tk.Button(frame, text=text, command=cmd,
                      bg="#555", fg="white", width=25).pack(pady=2)

    def browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp")]
        )
        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Cannot load image")
                return
            self.original_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_cv = self.original_cv.copy()
            self.display_image(self.current_cv)
            self.status_label.config(text=os.path.basename(path))

    def display_image(self, img):
        h, w = img.shape[:2]
        scale = min(700 / w, 600 / h, 1)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def add_noise_dialog(self, noise_type):
        dialog = tk.Toplevel(self.root)
        dialog.title("Noise Amount")

        tk.Label(dialog, text="Percentage (0-100)").pack()
        entry = tk.Entry(dialog)
        entry.insert(0, "10")
        entry.pack()

        def apply():
            p = float(entry.get()) / 100
            self.add_noise(noise_type, p)
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply).pack(pady=5)

    def add_noise(self, noise_type, percent):
        img = self.current_cv.copy()
        h, w = img.shape[:2]
        n = int(h * w * percent)

        for _ in range(n):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            if noise_type in ["salt", "both"]:
                img[x, y] = 255
            if noise_type in ["pepper", "both"]:
                img[x, y] = 0

        self.current_cv = img
        self.display_image(img)

    def apply_filter_dialog(self, ftype):
        dialog = tk.Toplevel(self.root)
        tk.Label(dialog, text="Mask size (odd)").pack()
        entry = tk.Entry(dialog)
        entry.insert(0, "3")
        entry.pack()

        boost_entry = None
        if ftype == "highboost":
            tk.Label(dialog, text="Boost").pack()
            boost_entry = tk.Entry(dialog)
            boost_entry.insert(0, "2")
            boost_entry.pack()

        def apply():
            k = int(entry.get()) | 1
            if ftype == "highboost":
                self.apply_high_boost(k, float(boost_entry.get()))
            elif ftype == "unsharp":
                self.apply_unsharp_mask(k)
            else:
                self.apply_morphological(ftype, k)
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply).pack(pady=5)

    def apply_morphological(self, ftype, k):
        gray = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        if ftype == "min":
            res = cv2.erode(gray, np.ones((k, k)))
        elif ftype == "max":
            res = cv2.dilate(gray, np.ones((k, k)))
        elif ftype == "median":
            res = cv2.medianBlur(gray, k)
        else:
            res = cv2.blur(gray, (k, k))

        self.current_cv = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def apply_edge_filter(self, ftype):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)

        if ftype == "laplacian":
            r = cv2.Laplacian(g, cv2.CV_64F)
        elif ftype == "sobel":
            sx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
            sy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
            r = np.sqrt(sx**2 + sy**2)
        elif ftype == "prewitt":
            kx = np.array([[-1,0,1]]*3)
            ky = kx.T
            r = np.sqrt(
                cv2.filter2D(g, -1, kx)**2 +
                cv2.filter2D(g, -1, ky)**2
            )
        else:
            kx = np.array([[1,0],[0,-1]])
            ky = np.array([[0,1],[-1,0]])
            r = np.sqrt(
                cv2.filter2D(g, -1, kx)**2 +
                cv2.filter2D(g, -1, ky)**2
            )

        r = np.uint8(np.clip(r, 0, 255))
        self.current_cv = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def apply_high_boost(self, k, a):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        blur = cv2.blur(g, (k, k))
        res = np.clip(g + a * (g - blur), 0, 255)
        self.current_cv = cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def apply_unsharp_mask(self, k):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(g, (k, k), 1)
        res = np.clip(g + (g - blur), 0, 255)
        self.current_cv = cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def reset_image(self):
        if self.original_cv is not None:
            self.current_cv = self.original_cv.copy()
            self.display_image(self.current_cv)

    def save_image(self):
        if self.current_cv is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        cv2.imwrite(path, cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()