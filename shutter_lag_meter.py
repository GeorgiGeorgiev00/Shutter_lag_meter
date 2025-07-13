import cv2
import numpy as np
import subprocess
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ADB_IP = "192.168.4.13"
DESTINATION_FOLDER = "/home/user/Desktop/diplomna"

def adb_connect(ip_address=ADB_IP):
    print(f"Connecting to ADB device @ {ip_address}...")
    try:
        subprocess.run(["adb", "connect", ip_address], check=True)
        print("Connected successfully.")
    except subprocess.CalledProcessError as e:
        print("ADB connection failed:", e)

def pull_last_files (destination_folder, source_folder="/sdcard/DCIM/Camera/", n=100):
    print(f"Pulling the last {n} files from the phone...")
    try:
        result = subprocess.run(
            ["adb", "shell", f"ls -t {source_folder}"],
            capture_output=True, text=True, check=True
        )
        files = result.stdout.strip().split('\n')[:n]
        if not files:
            print("No files found on the phone.")
            return
        os.makedirs(destination_folder, exist_ok=True)
        for file in files:
            local_path = os.path.join(destination_folder, file)
            print(f"Downloading: {file}")
            subprocess.run(["adb", "pull", f"{source_folder}{file}", local_path])
    except subprocess.CalledProcessError as e:
        print("ADB error:", e)


class ROISelector:
    def __init__(self, image, scale_factor=0.5):
        self.scale_factor = scale_factor
        self.image = self.resize_image(image)
        self.rois = []
        self.current_roi = None
        self.drawing = False

    def resize_image(self, image):
        if self.scale_factor != 1.0:
            width = int(image.shape[1] * self.scale_factor)
            height = int(image.shape[0] * self.scale_factor)
            return cv2.resize(image, (width, height))
        return image.copy()

    def select_rois(self):
        cv2.namedWindow("Select 9 regions", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select 9 regions", self.mouse_callback)

        while True:
            img = self.image.copy()
            for i, (x, y, w, h) in enumerate(self.rois):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str(i + 1), (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.current_roi:
                x, y, w, h = self.current_roi
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Select 9 regions", img)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter
                if self.current_roi:
                    self.rois.append(self.current_roi)
                    self.current_roi = None
                    if len(self.rois) >= 9:
                        break
            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()

        if len(self.rois) == 9:
            original_rois = []
            for x, y, w, h in self.rois:
                orig_x = int(x / self.scale_factor)
                orig_y = int(y / self.scale_factor)
                orig_w = int(w / self.scale_factor)
                orig_h = int(h / self.scale_factor)
                original_rois.append((orig_x, orig_y, orig_w, orig_h))
            return original_rois
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_roi = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current_roi is not None:
                self.current_roi[2] = x - self.current_roi[0]
                self.current_roi[3] = y - self.current_roi[1]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_roi is not None:
                if self.current_roi[2] < 0:
                    self.current_roi[0] += self.current_roi[2]
                    self.current_roi[2] *= -1
                if self.current_roi[3] < 0:
                    self.current_roi[1] += self.current_roi[3]
                    self.current_roi[3] *= -1

def analyze_regions(image, rois):
    results = []
    MIN_BRIGHTNESS = 180
    RED_MARGIN = 20
    GREEN_MARGIN = 20
    BLUE_MARGIN = 5

    for i, roi in enumerate(rois):
        x, y, w, h = roi
        region = image[y:y + h, x:x + w]
        avg_color = np.mean(region, axis=(0, 1))
        b, g, r = avg_color

        brightness = max(b, g, r)

        if brightness < MIN_BRIGHTNESS:
            color_name = "No dominant color"
            diode_num = 0
        elif (r - max(g, b) > RED_MARGIN):
            color_name = "Red"
            diode_num = i + 19
        elif (g - max(r, b) > GREEN_MARGIN):
            color_name = "Green"
            diode_num = i + 10
        elif (b > r) and (b > g) and (b - min(r, g) > BLUE_MARGIN):
            color_name = "Blue"
            diode_num = i + 1
        else:
            color_name = "No dominant color"
            diode_num = 0

        results.append((diode_num, color_name, b, g, r, brightness))
    return results


def calculate_diode_times(results, time_step, zero_diode):
    return [(diode_num, 0 if diode_num == 0 else (diode_num - zero_diode) * time_step) for diode_num, *_ in results]


def full_results(path, image_file, results, diode_times):
    with open(path, "a") as f:
        f.write(f"Image: {image_file}\n")
        for (diode_num, color_name, b, g, r,brightness), (_, t) in zip(results, diode_times):
            if diode_num == 0:
                f.write(f"  Diode: None active, BGR=({b:.1f}, {g:.1f}, {r:.1f}), Brightness={brightness:.1f}, Color: {color_name}, Time: 0 ms\n")
            else:
                f.write(f"  Diode {diode_num}: BGR=({b:.1f}, {g:.1f}, {r:.1f}), Brightness={brightness:.1f}, Color: {color_name}, Time: {t} ms\n")
        f.write("\n")

def write_closest_diode(path, image_file, results, diode_times, zero_diode):
    candidates = []
    for (diode_num, color_name, _, _, _, _), (_, t) in zip(results, diode_times):
        if color_name == "No dominant color":
            continue
        candidates.append((diode_num, t))

    sel_diode, sel_time = None, None

    if candidates:
        neg_candidates = [c for c in candidates if c[1] < 0]
        pos_candidates = [c for c in candidates if c[1] >= 0]

        if neg_candidates:
            sel_diode, sel_time = min(neg_candidates, key=lambda x: x[1])
        elif pos_candidates:
            sel_diode, sel_time = min(pos_candidates, key=lambda x: abs(x[1]))  

    with open(path, "a") as f:
        if sel_diode is not None:
            f.write(f"{image_file}: Diode {sel_diode}, Time {sel_time} ms\n")
            return sel_time
        else:
            f.write(f"{image_file}: No active diode with valid color found\n")
            return None

def write_statistics(path, times):
    with open(path, "a") as f:
        f.write("\n--- Statistics ---\n")
        f.write(f"Average time: {sum(times)/len(times):.2f} ms\n")
        f.write(f"Minimum time: {min(times)} ms\n")
        f.write(f"Maximum time: {max(times)} ms\n")

def histogram(times_file, output_img_path="histogram_times.png"):
    if not os.path.exists(times_file):
        print(f"File {times_file} not found, skipping histogram creation.")
        return

    with open(times_file, "r", encoding="utf-8") as f:
        text = f.read()

    times = re.findall(r'Time\s+(-?\d+)\s+ms', text)
    times = list(map(int, times))
    if not times:
        print("No time values found in the file.")
        return

    counter = Counter(times)
    sorted_times = sorted(counter.items())

    x_vals = [item[0] for item in sorted_times]
    y_vals = [item[1] for item in sorted_times]

    plt.figure(figsize=(10, 5))
    bar_width = 1
    plt.bar(x_vals, y_vals, width=bar_width, color="royalblue", edgecolor='black')

    plt.xticks(x_vals, rotation=45)  

    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    average_time = sum(times) / len(times)
    plt.text(min(x_vals), max(y_vals) * 0.95, f"Average time: {average_time:.2f} ms",
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    plt.title("Distribution of capture times (ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of captures")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    print(f"Histogram saved at: {output_img_path}")


def analyze_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_files:
        print("No images found.")
        return

    while True:
        try:
            time_step = int(input("Enter time step (ms): "))
            break
        except ValueError:
            print("Please enter a valid integer for time step.")

    while True:
        try:
            zero_diode = int(input("Enter zero diode number (1-27): "))
            if 1 <= zero_diode <= 27:
                break
            else:
                print("Diode number must be between 1 and 27.")
        except ValueError:
            print("Please enter a valid integer between 1 and 27.")

    full_out = os.path.join(folder_path, "full_results.txt")
    closest_out = os.path.join(folder_path, "closest_diode_times.txt")
    times = []

    rois = None
    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(folder_path, image_file)
        print(f"\nProcessing: {image_file}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Cannot load image: {image_file}")
            continue

        if idx == 0:
            selector = ROISelector(image)
            rois = selector.select_rois()
            if not rois or len(rois) != 9:
                print("You must select exactly 9 regions.")
                return

        results = analyze_regions(image, rois)
        diode_times = calculate_diode_times(results, time_step, zero_diode)
        full_results(full_out, image_file, results, diode_times)
        t = write_closest_diode(closest_out, image_file, results, diode_times, zero_diode)
        if t is not None:
            times.append(t)

    if times:
        write_statistics(closest_out, times)
    else:
        print("No diode times found to calculate statistics.")

if __name__ == "__main__":
    adb_connect()
    pull_last_files(DESTINATION_FOLDER)

    analyze_images_in_folder(DESTINATION_FOLDER)

    histogram_image_path = os.path.join(DESTINATION_FOLDER, "closest_diode_times_histogram.png")
    closest_times_file = os.path.join(DESTINATION_FOLDER, "closest_diode_times.txt")
    histogram(closest_times_file, histogram_image_path)
