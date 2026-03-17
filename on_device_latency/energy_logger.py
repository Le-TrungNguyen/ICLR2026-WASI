import os
import glob
import time
import threading
from collections import defaultdict

class EnergyLogger:
    """
    Đo năng lượng theo từng phase (ví dụ: 'train_forward', 'train_backward', 'inference')
    bằng cách đọc cảm biến INA3221 trên Jetson Orin.

    - Mặc định dùng rail 'VDD_CPU_GPU_CV' (CPU+GPU) nếu có, nếu không thì fallback sang 'VDD_IN'.
    - Năng lượng đơn vị: Joule (J).
    """

    def __init__(self, interval=0.05, rails_to_use=None):
        """
        interval: thời gian giữa các lần đọc sensor (giây)
        rails_to_use: list tên rail, ví dụ ["VDD_CPU_GPU_CV"] hoặc ["VDD_IN"]
                      Nếu None: ưu tiên VDD_CPU_GPU_CV, fallback VDD_IN.
        """
        self.interval = interval
        self.hwmon = self._find_hwmon_path()
        all_rails = self._find_rails(self.hwmon)

        if rails_to_use is None:
            names = [r[0] for r in all_rails]
            rails = []
            if "VDD_CPU_GPU_CV" in names:
                rails.append(all_rails[names.index("VDD_CPU_GPU_CV")])
            if "VDD_IN" in names:
                rails.append(all_rails[names.index("VDD_IN")])
            if not rails:
                # nếu không tìm được, lấy tất cả
                rails = all_rails
        else:
            names = [r[0] for r in all_rails]
            rails = []
            for name in rails_to_use:
                if name in names:
                    rails.append(all_rails[names.index(name)])
            if not rails:
                raise ValueError(f"Không tìm thấy rail nào trong {rails_to_use}. Có các rail: {names}")

        self.rails = rails   # list[(label, idx, curr_path, volt_path)]

        # năng lượng tích luỹ theo phase và rail: energy[phase][label] = Joules
        self.energy = defaultdict(lambda: defaultdict(float))
        self.time = defaultdict(float)

        self.current_phase = None
        self.paused = False           # <--- thêm cờ pause
        self.lock = threading.Lock()
        self.stop_flag = False
        self.thread = None

    # ---------- sysfs helpers ----------

    def _find_hwmon_path(self):

        # Thử lần lượt các driver có thể có trên Jetson (Orin + Nano)
        patterns = [
            "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*",   # Orin, Xavier,...
            "/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*",  # Nano (như máy bạn)
            "/sys/bus/i2c/drivers/ina230x/*/hwmon/hwmon*",
            "/sys/bus/i2c/drivers/ina219x/*/hwmon/hwmon*",
        ]

        for pat in patterns:
            candidates = glob.glob(pat)
            if candidates:
                return candidates[0]
            
        raise RuntimeError("Không tìm thấy hwmon cho INA sensor (ina3221/ina3221x/ina230x/ina219x).")

        # candidates = glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")
        # if not candidates:
        #     raise RuntimeError("Không tìm thấy hwmon cho ina3221.")
        # return candidates[0]

    def _find_rails(self, hwmon_path):
        rails = []
        for i in range(1, 9):
            curr_path = os.path.join(hwmon_path, f"curr{i}_input")
            volt_path = os.path.join(hwmon_path, f"in{i}_input")
            if os.path.exists(curr_path) and os.path.exists(volt_path):
                label_path = os.path.join(hwmon_path, f"in{i}_label")
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        label = f.read().strip()
                else:
                    label = f"rail{i}"
                rails.append((label, i, curr_path, volt_path))
        if not rails:
            raise RuntimeError("Không tìm thấy rail nào có cả currN_input và inN_input.")
        return rails

    def _read_int(self, path):
        with open(path, "r") as f:
            return int(f.read().strip())

    # ---------- sampler thread ----------

    def _sampler_loop(self):
        last_t = time.time()
        while not self.stop_flag:
            now = time.time()
            dt = now - last_t
            last_t = now

            with self.lock:
                phase = self.current_phase
                paused = self.paused

            # chỉ cộng năng lượng khi:
            # - đang ở một phase cụ thể
            # - và KHÔNG bị pause
            if phase is not None and not paused:
                # CỘNG TIME CHO PHASE
                self.time[phase] += dt  


                for label, idx, curr_path, volt_path in self.rails:
                    curr_ma = self._read_int(curr_path)   # mA
                    volt_mv = self._read_int(volt_path)   # mV
                    power_w = (curr_ma * volt_mv) / 1_000_000.0  # mA*mV -> W
                    self.energy[phase][label] += power_w * dt    # J = W * s

            time.sleep(self.interval)

    # ---------- public API ----------

    def start_global(self):
        """Bắt đầu thread sampler (chạy một lần trước training/inference)."""
        if self.thread is not None:
            return
        self.stop_flag = False
        self.thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self.thread.start()

    def stop_global(self):
        """Dừng sampler (gọi sau khi toàn bộ training+inference xong)."""
        if self.thread is None:
            return
        self.stop_flag = True
        self.thread.join()
        self.thread = None

    def start_phase(self, name):
        """Bắt đầu một phase (vd: 'train_forward', 'train_backward', 'inference')."""
        with self.lock:
            self.current_phase = name
            self.paused = False   # đảm bảo phase mới không bị pause

    def stop_phase(self):
        """Dừng phase hiện tại (không tắt sampler)."""
        with self.lock:
            self.current_phase = None
            self.paused = False

    def pause(self):
        """Tạm dừng đo năng lượng trong phase hiện tại (vd: skip SVD init)."""
        with self.lock:
            self.paused = True

    def resume(self):
        """Tiếp tục đo năng lượng trong phase hiện tại."""
        with self.lock:
            self.paused = False

    def get_energy(self):
        """
        Trả về dict {phase: {rail_label: energy_J}}.
        """
        return self.energy
    
    def get_time(self):
        """Trả về dict {phase: time_seconds}."""
        return self.time
