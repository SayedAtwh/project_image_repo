import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# ================== THEME ==================
BG_MAIN = "#1e1e2f"
BG_PANEL = "#2a2a40"
BTN_COLOR = "#4CAF50"
BTN_HOVER = "#66BB6A"
TEXT_COLOR = "white"

APP_FONT = ("Segoe UI", 10)
TITLE_FONT = ("Segoe UI", 13, "bold")


class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.root.geometry("1250x800")
        self.root.configure(bg=BG_MAIN)

        self.original_cv = None
        self.current_cv = None

        self.create_widgets()

    # ========== UI HELPERS ==========
    def styled_button(self, parent, text, cmd, color=BTN_COLOR):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white",
            font=APP_FONT,
            relief="flat",
            width=14
        )
        btn.pack(side=tk.LEFT, padx=6, pady=10)
        btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    # ========== UI ==========
    def create_widgets(self):
        # ---------- TOP BAR ----------
        top_frame = tk.Frame(self.root, bg=BG_PANEL, height=70)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        top_frame.pack_propagate(False)

        tk.Label(
            top_frame, text="Image Processing Tool",
            bg=BG_PANEL, fg=TEXT_COLOR,
            font=TITLE_FONT
        ).pack(side=tk.LEFT, padx=15)

        self.styled_button(top_frame, "📁 Browse", self.browse_image)
        self.styled_button(top_frame, "🔄 Reset", self.reset_image, "#FF9800")
        self.styled_button(top_frame, "💾 Save", self.save_image, "#2196F3")

        # ---------- MAIN CONTENT ----------
        content = tk.Frame(self.root, bg=BG_MAIN)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---------- IMAGE PANEL ----------
        image_frame = tk.Frame(content, bg=BG_PANEL, bd=2, relief="ridge")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.image_label = tk.Label(
            image_frame, bg="black", fg="gray",
            text="No Image Loaded", font=("Segoe UI", 14)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # ---------- FILTER PANEL ----------
        filter_frame = tk.Frame(content, bg=BG_PANEL, width=300)
        filter_frame.pack(side=tk.RIGHT, fill=tk.Y)
        filter_frame.pack_propagate(False)

        tk.Label(
            filter_frame, text="Filters & Effects",
            bg=BG_PANEL, fg=TEXT_COLOR,
            font=TITLE_FONT
        ).pack(pady=10)

        canvas = tk.Canvas(filter_frame, bg=BG_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(filter_frame, command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG_PANEL)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.create_section(scroll_frame, "NOISE", [
            ("Salt Noise", lambda: self.add_noise_dialog("salt")),
            ("Pepper Noise", lambda: self.add_noise_dialog("pepper")),
            ("Salt & Pepper", lambda: self.add_noise_dialog("both")),
        ])

        self.create_section(scroll_frame, "FILTERS", [
            ("Min Filter", lambda: self.apply_filter_dialog("min")),
            ("Max Filter", lambda: self.apply_filter_dialog("max")),
            ("Median Filter", lambda: self.apply_filter_dialog("median")),
            ("Mean Filter", lambda: self.apply_filter_dialog("mean")),
        ])

        self.create_section(scroll_frame, "EDGE DETECTION", [
            ("Laplacian", lambda: self.apply_edge_filter("laplacian")),
            ("Sobel", lambda: self.apply_edge_filter("sobel")),
            ("Prewitt", lambda: self.apply_edge_filter("prewitt")),
            ("Roberts", lambda: self.apply_edge_filter("roberts")),
        ])

        self.create_section(scroll_frame, "ENHANCEMENT", [
            ("High Boost", lambda: self.apply_filter_dialog("highboost")),
            ("Unsharp Mask", lambda: self.apply_filter_dialog("unsharp")),
        ])

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------- STATUS BAR ----------
        self.status_bar = tk.Label(
            self.root, text="Ready",
            bg=BG_PANEL, fg="white",
            anchor="w", padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_section(self, parent, title, buttons):
        card = tk.Frame(parent, bg=BG_MAIN, bd=1, relief="groove")
        card.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            card, text=title,
            bg=BG_MAIN, fg=TEXT_COLOR,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=8, pady=(5, 3))

        for text, cmd in buttons:
            tk.Button(
                card, text=text, command=cmd,
                bg="#3c3c5a", fg="white",
                relief="flat", font=APP_FONT, width=24
            ).pack(pady=3, padx=8)

    # ========== LOGIC ==========
    def browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot load image")
            return

        self.original_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_cv = self.original_cv.copy()
        self.display_image(self.current_cv)
        self.status_bar.config(text=f"Loaded: {os.path.basename(path)}")

    def display_image(self, img):
        h, w = img.shape[:2]
        scale = min(700 / w, 600 / h, 1)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def add_noise_dialog(self, ntype):
        if self.current_cv is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Noise Amount")
        dialog.configure(bg=BG_PANEL)

        tk.Label(dialog, text="Percentage (0-100)",
                 bg=BG_PANEL, fg="white").pack(pady=5)
        entry = tk.Entry(dialog)
        entry.insert(0, "10")
        entry.pack()

        def apply():
            self.add_noise(ntype, float(entry.get()) / 100)
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply).pack(pady=10)

    def add_noise(self, ntype, p):
        img = self.current_cv.copy()
        h, w = img.shape[:2]
        n = int(h * w * p)

        for _ in range(n):
            x, y = np.random.randint(0, h), np.random.randint(0, w)
            if ntype in ["salt", "both"]:
                img[x, y] = 255
            if ntype in ["pepper", "both"]:
                img[x, y] = 0

        self.current_cv = img
        self.display_image(img)
        self.status_bar.config(text="Noise applied")

    def apply_filter_dialog(self, ftype):
        if self.current_cv is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Filter Settings")

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

        tk.Button(dialog, text="Apply", command=apply).pack(pady=10)

    def apply_morphological(self, ftype, k):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        if ftype == "min":
            r = cv2.erode(g, np.ones((k, k)))
        elif ftype == "max":
            r = cv2.dilate(g, np.ones((k, k)))
        elif ftype == "median":
            r = cv2.medianBlur(g, k)
        else:
            r = cv2.blur(g, (k, k))

        self.current_cv = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
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
            r = np.sqrt(cv2.filter2D(g, -1, kx)**2 +
                        cv2.filter2D(g, -1, ky)**2)
        else:
            kx = np.array([[1,0],[0,-1]])
            ky = np.array([[0,1],[-1,0]])
            r = np.sqrt(cv2.filter2D(g, -1, kx)**2 +
                        cv2.filter2D(g, -1, ky)**2)

        r = np.uint8(np.clip(r, 0, 255))
        self.current_cv = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def apply_high_boost(self, k, a):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        blur = cv2.blur(g, (k, k))
        r = np.clip(g + a * (g - blur), 0, 255)
        self.current_cv = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def apply_unsharp_mask(self, k):
        g = cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(g, (k, k), 1)
        r = np.clip(g + (g - blur), 0, 255)
        self.current_cv = cv2.cvtColor(r.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.display_image(self.current_cv)

    def reset_image(self):
        if self.original_cv is not None:
            self.current_cv = self.original_cv.copy()
            self.display_image(self.current_cv)
            self.status_bar.config(text="Image reset")

    def save_image(self):
        if self.current_cv is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        cv2.imwrite(path, cv2.cvtColor(self.current_cv, cv2.COLOR_RGB2BGR))
        self.status_bar.config(text="Image saved")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()
