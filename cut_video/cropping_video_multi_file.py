import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import threading

######
# 裁剪视频画面，批量裁剪文件夹内的所有视频
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
        self.frame_rate = 0
        self.scale = None
        self.folder_path = None
        self.crop_ratio = None
        self.progress_single = None
        self.progress_total = None
        self.lock = threading.Lock()  # 添加线程锁

        self.open_button = tk.Button(self.window, text="Open", command=self.open_folder)
        self.open_button.pack()

        self.crop_button = tk.Button(self.window, text="Crop", command=self.crop_videos)
        self.crop_button.pack()

        # 进度条
        self.progress_single = ttk.Progressbar(self.window, orient="horizontal", length=200, mode="determinate")
        self.progress_single.pack()

        self.progress_total = ttk.Progressbar(self.window, orient="horizontal", length=200, mode="determinate")
        self.progress_total.pack()

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

    def open_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            video_files = glob.glob(os.path.join(self.folder_path, "*.mp4")) + glob.glob(
                os.path.join(self.folder_path, "*.avi"))
            if video_files:
                self.open_video(video_files[0])
            else:
                messagebox.showwarning("Warning", "No video files found in the selected folder.")

    def open_video(self, video_path):
        self.video_path = video_path
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
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
                self.scale = tk.Scale(self.window, from_=0, to=self.frame_count - 1, orient=tk.HORIZONTAL,
                                      command=self.update_frame)
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
            if self.lock_ratio_var.get():
                ratio = [int(r) for r in self.ratio_var.get().split(":")]
                width = self.crop_x2 - self.crop_x1
                height = int(width / ratio[0] * ratio[1])
                self.crop_y2 = self.crop_y1 + height
            else:
                self.crop_y2 = event.y
            self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2,
                                                     outline="red")
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

    def crop_videos(self):
        if self.video_path and self.frame is not None:
            x1 = min(self.crop_x1, self.crop_x2)
            y1 = min(self.crop_y1, self.crop_y2)
            x2 = max(self.crop_x1, self.crop_x2)
            y2 = max(self.crop_y1, self.crop_y2)
            if x2 - x1 > 0 and y2 - y1 > 0:
                ratio_str = self.ratio_var.get().replace(":", "-")
                video_files = glob.glob(os.path.join(self.folder_path, "*.mp4")) + glob.glob(
                    os.path.join(self.folder_path, "*.avi"))
                total_videos = len(video_files)
                self.progress_total['maximum'] = total_videos
                # 创建线程池
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for video_file in video_files:
                        future = executor.submit(self.crop_video, video_file, x1, y1, x2, y2, ratio_str, total_videos)
                        futures.append(future)
                    # 在另一个线程中更新进度条
                    threading.Thread(target=self.update_progress, args=(futures,)).start()
            else:
                messagebox.showwarning("Warning", "Invalid crop area selected!")
        else:
            messagebox.showwarning("Warning", "No video loaded or frame not selected!")

    def crop_video(self, video_path, x1, y1, x2, y2, ratio_str, total_videos):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        folder_name = os.path.join(os.path.dirname(video_path), "crop")
        os.makedirs(folder_name, exist_ok=True)  # 确保crop子目录存在
        out = cv2.VideoWriter(os.path.join(folder_name, os.path.splitext(os.path.basename(video_path))[
            0] + "_crop_" + ratio_str + ".mp4"), fourcc, self.frame_rate, (x2 - x1, y2 - y1))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[y1:y2, x1:x2]
            out.write(frame)

        cap.release()
        out.release()

    def update_progress(self, futures):
        # 检查任务是否完成，并更新进度条
        def check_progress():
            done_count = sum(future.done() for future in futures)
            self.progress_total['value'] = done_count
            if done_count < len(futures):
                self.window.after(100, check_progress)
            else:
                messagebox.showinfo("Info", "All videos cropped successfully!")

        check_progress()

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    cropper = VideoCropper()
    cropper.run()
