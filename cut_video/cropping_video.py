import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
######
# 裁剪视频画面，裁剪单个视频
######
class VideoCropper:
    def __init__(self):
        self.dragging_inside = None
        self.drag_start_y = None
        self.drag_start_x = None
        self.window = tk.Tk()
        self.window.title("Video Cropper")
        self.canvas = None
        self.rect = None
        self.video_path = None
        self.frame = None
        self.crop_x1 = 0
        self.crop_y1 = 0
        self.crop_x2 = 100
        self.crop_y2 = 100
        self.dragging = False
        self.width = 500
        self.height = 500
        self.cap = None
        self.frame_count = 0
        self.scale = None

        self.open_button = tk.Button(self.window, text="Open", command=self.open_video)
        self.open_button.pack()

        self.crop_button = tk.Button(self.window, text="Crop", command=self.crop_video)
        self.crop_button.pack()

        # 进度条
        self.progress = ttk.Progressbar(self.window, orient="horizontal", length=200, mode="determinate")
        self.progress.pack()

        # 比例锁定复选框
        self.lock_ratio_var = tk.IntVar()
        self.lock_ratio_check = tk.Checkbutton(self.window, text="Lock Ratio", variable=self.lock_ratio_var)
        self.lock_ratio_check.pack()

        # 比例选择
        self.ratio_var = tk.StringVar()
        self.ratio_var.set("1:1")
        self.ratio_choices = ["1:1", "4:3", "16:9", "9:16"]
        self.ratio_option = tk.OptionMenu(self.window, self.ratio_var, *self.ratio_choices)
        self.ratio_option.pack()

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
                self.canvas.pack()
                self.canvas.bind("<Button-1>", self.start_drag)
                self.canvas.bind("<B1-Motion>", self.do_drag)
                self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
                self.scale = tk.Scale(self.window, from_=0, to=self.frame_count-1, orient=tk.HORIZONTAL, command=self.update_frame)
                self.scale.pack(fill=tk.X)
                self.show_frame()

    def show_frame(self):
        image = Image.fromarray(self.frame)
        image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.canvas.image = image_tk
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2, outline="red")

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        if self.crop_x1 < event.x < self.crop_x2 and self.crop_y1 < event.y < self.crop_y2:
            self.dragging_inside = True
        else:
            self.dragging = True

    def do_drag(self, event):
        if self.dragging:
            self.crop_x2 = event.x
            if self.lock_ratio_var.get():  # 如果启用了比例锁定
                ratio = [int(r) for r in self.ratio_var.get().split(":")]
                width = self.crop_x2 - self.crop_x1
                height = int(width / ratio[0] * ratio[1])
                self.crop_y2 = self.crop_y1 + height
            else:
                self.crop_y2 = event.y
            self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2, outline="red")
        elif self.dragging_inside:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.crop_x1 += dx
            self.crop_y1 += dy
            self.crop_x2 += dx
            self.crop_y2 += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2,
                                                     outline="red")

    def stop_drag(self, event):
        self.dragging = False
        self.dragging_inside = False

    def update_frame(self, value):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
        ret, frame = self.cap.read()
        if ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame()

    def crop_video(self):
        if self.video_path and self.frame is not None:
            x1 = min(self.crop_x1, self.crop_x2)
            y1 = min(self.crop_y1, self.crop_y2)
            x2 = max(self.crop_x1, self.crop_x2)
            y2 = max(self.crop_y1, self.crop_y2)
            if x2 - x1 > 0 and y2 - y1 > 0:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (x2 - x1, y2 - y1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.progress['maximum'] = self.frame_count
                for _ in range(self.frame_count):
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame = frame[y1:y2, x1:x2]
                    out.write(frame)
                    self.progress['value'] += 1
                    self.window.update_idletasks()
                out.release()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.progress['value'] = 0
                messagebox.showinfo("Info", "Video cropped successfully!")
            else:
                messagebox.showwarning("Warning", "Invalid crop area selected!")
        else:
            messagebox.showwarning("Warning", "No video loaded or frame not selected!")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    cropper = VideoCropper()
    cropper.run()
