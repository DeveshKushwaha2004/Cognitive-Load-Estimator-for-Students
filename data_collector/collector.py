"""
Data Collector for Cognitive Load Estimator.

Runs locally on the student's machine to capture:
- Keyboard events (typing speed, backspace rate)
- Mouse events (movement distance, jitter)
- Tab-switch detection

Sends aggregated feature data every 5 seconds to the backend API.

Requirements (install locally):
    pip install pynput pygetwindow requests

Usage:
    python collector.py --server http://localhost:8000 --token YOUR_JWT_TOKEN
"""

import time
import math
import argparse
import threading
import requests

try:
    from pynput import keyboard, mouse
except ImportError:
    print("ERROR: pynput is required. Install with: pip install pynput")
    raise

try:
    import pygetwindow as gw
except ImportError:
    gw = None
    print("WARNING: pygetwindow not available. Tab-switch tracking disabled.")
    print("  Install with: pip install pygetwindow")


class DataCollector:
    """Collects keyboard and mouse behavioral data in real time."""

    def __init__(self, server_url: str, token: str, interval: float = 5.0):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.interval = interval

        # Keyboard tracking
        self.key_count = 0
        self.backspace_count = 0
        self.key_timestamps: list[float] = []

        # Mouse tracking
        self.mouse_positions: list[tuple[float, float]] = []
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Tab-switch tracking
        self.active_window = ""
        self.tab_switches = 0

        self._lock = threading.Lock()

    def on_key_press(self, key):
        with self._lock:
            self.key_count += 1
            self.key_timestamps.append(time.time())
            if key == keyboard.Key.backspace:
                self.backspace_count += 1

    def on_mouse_move(self, x, y):
        with self._lock:
            self.mouse_positions.append((x, y))

    def check_tab_switch(self):
        if gw is None:
            return
        try:
            active = gw.getActiveWindow()
            title = active.title if active else ""
            if title != self.active_window and self.active_window:
                self.tab_switches += 1
            self.active_window = title
        except Exception:
            pass

    def compute_features(self) -> dict:
        with self._lock:
            # Typing speed (keys per second)
            typing_speed = self.key_count / self.interval

            # Speed variance
            if len(self.key_timestamps) > 1:
                intervals = [
                    self.key_timestamps[i] - self.key_timestamps[i - 1]
                    for i in range(1, len(self.key_timestamps))
                ]
                mean_interval = sum(intervals) / len(intervals)
                speed_variance = (
                    sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                )
            else:
                speed_variance = 0.0

            # Backspace rate
            backspace_rate = (
                self.backspace_count / self.key_count if self.key_count > 0 else 0.0
            )

            # Mouse distance
            mouse_distance = 0.0
            for i in range(1, len(self.mouse_positions)):
                x1, y1 = self.mouse_positions[i - 1]
                x2, y2 = self.mouse_positions[i]
                mouse_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Mouse jitter (direction changes)
            mouse_jitter = 0.0
            if len(self.mouse_positions) > 2:
                for i in range(2, len(self.mouse_positions)):
                    x0, y0 = self.mouse_positions[i - 2]
                    x1, y1 = self.mouse_positions[i - 1]
                    x2, y2 = self.mouse_positions[i]
                    dx1, dy1 = x1 - x0, y1 - y0
                    dx2, dy2 = x2 - x1, y2 - y1
                    cross = abs(dx1 * dy2 - dy1 * dx2)
                    if cross > 10:  # Threshold for direction change
                        mouse_jitter += 1

            tab_switch_count = float(self.tab_switches)

            # Reset counters
            self.key_count = 0
            self.backspace_count = 0
            self.key_timestamps.clear()
            self.mouse_positions.clear()
            self.tab_switches = 0

        return {
            "typing_speed": round(typing_speed, 4),
            "speed_variance": round(speed_variance, 6),
            "backspace_rate": round(backspace_rate, 4),
            "mouse_distance": round(mouse_distance, 2),
            "mouse_jitter": round(mouse_jitter, 2),
            "tab_switch_count": round(tab_switch_count, 2),
        }

    def send_data(self, features: dict):
        try:
            resp = requests.post(
                f"{self.server_url}/api/cognitive/log",
                json=features,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=5,
            )
            if resp.status_code == 200:
                result = resp.json()
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Load: {result.get('predicted_load', 'N/A'):6s} | "
                    f"Speed: {features['typing_speed']:.2f} keys/s | "
                    f"Backspace: {features['backspace_rate']:.2%} | "
                    f"Mouse: {features['mouse_distance']:.0f}px | "
                    f"Jitter: {features['mouse_jitter']:.0f}"
                )
            else:
                print(f"[ERROR] Server returned {resp.status_code}: {resp.text}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to send data: {e}")

    def run(self):
        print("=" * 60)
        print("Cognitive Load Data Collector")
        print("=" * 60)
        print(f"Server: {self.server_url}")
        print(f"Interval: {self.interval}s")
        print("Press Ctrl+C to stop.\n")

        # Start listeners
        kb_listener = keyboard.Listener(on_press=self.on_key_press)
        mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        kb_listener.start()
        mouse_listener.start()

        try:
            while True:
                time.sleep(self.interval)
                self.check_tab_switch()
                features = self.compute_features()
                self.send_data(features)
        except KeyboardInterrupt:
            print("\nStopping data collection...")
        finally:
            kb_listener.stop()
            mouse_listener.stop()


def main():
    parser = argparse.ArgumentParser(description="Cognitive Load Data Collector")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="JWT authentication token (from login)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Data collection interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    collector = DataCollector(
        server_url=args.server,
        token=args.token,
        interval=args.interval,
    )
    collector.run()


if __name__ == "__main__":
    main()
