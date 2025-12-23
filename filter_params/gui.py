from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QLabel,
    QApplication, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)
from matplotlib.figure import Figure

import numpy as np

from filters.kalman import KalmanFilter
from signals.generate import normally_noisy
from filter_params.metrics import metrics


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class KalmanWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalman Filter Demo (PyQt6 + Matplotlib)")

        self.canvas = MatplotlibCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # ===== TOP sliders =====
        self.top_sliders = {
            "Noise": self.make_slider_horizontal(1, 100, 15, "Noise %"),
            "Seed": self.make_slider_horizontal(0, 9999, 42, "Random Seed"),
        }

        # ===== BOTTOM sliders =====
        self.sliders = {
            "R": self.make_slider_horizontal(1, 3000, 1000, "R (measurement noise)"),
            "Q": self.make_slider_horizontal(0, 500, 0, "Q (process noise) /1000"),
            "A": self.make_slider_horizontal(50, 150, 100, "A /100"),
            "H": self.make_slider_horizontal(50, 150, 100, "H /100"),
        }

        reset_btn = QPushButton("Reset sliders")
        reset_btn.clicked.connect(self.reset_values)

        # ===== Metrics label =====
        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet(
            "font-family: monospace; font-size: 11px;"
        )

        # ===== Layout =====
        layout = QVBoxLayout()

        for s in self.top_sliders.values():
            layout.addLayout(s["layout"])

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.metrics_label)

        for s in self.sliders.values():
            layout.addLayout(s["layout"])

        layout.addWidget(reset_btn)
        self.setLayout(layout)

        for s in list(self.top_sliders.values()) + list(self.sliders.values()):
            s["slider"].valueChanged.connect(self.update_plot)

        self.filter = KalmanFilter(
            1,
            1,
            1,
            1,
            0,
            1
        )

        self.update_plot()

    def make_slider_horizontal(self, mn, mx, val, name):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(mn)
        slider.setMaximum(mx)
        slider.setValue(val)

        label = QLabel(f"{name}: {val}")
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(slider)

        return {
            "slider": slider,
            "label": label,
            "layout": layout,
            "name": name,
        }

    def reset_values(self):
        defaults = dict(R=1000, Q=0, A=100, H=100)
        for name, params in self.sliders.items():
            params["slider"].setValue(defaults[name])

        self.top_sliders["Noise"]["slider"].setValue(15)
        self.top_sliders["Seed"]["slider"].setValue(42)

        self.update_plot()

    def update_plot(self):
        # ===== Top settings =====
        noise = self.top_sliders["Noise"]["slider"].value() / 100
        seed = self.top_sliders["Seed"]["slider"].value()
        np.random.seed(seed)

        # ===== Bottom settings =====
        R = self.sliders["R"]["slider"].value()
        Q = self.sliders["Q"]["slider"].value() / 1000
        A = self.sliders["A"]["slider"].value() / 100
        H = self.sliders["H"]["slider"].value() / 100

        # Update labels
        for s in list(self.top_sliders.values()) + list(self.sliders.values()):
            s["label"].setText(f"{s['name']}: {s['slider'].value()}")

        # ===== Signal & filtering =====
        t, clear, noisy = normally_noisy(np.sin, 0, 6 * np.pi, 300, noise)

        # with open("data.csv", "w") as f:
        #     for vals in zip(t, clear, noisy):
        #         print(*vals, sep=";", file=f)

        self.filter.Q = Q
        self.filter.R = R
        self.filter.A = A
        self.filter.H = H

        filtered = self.filter(noisy)

        # ===== Plot =====
        ax = self.canvas.ax
        ax.clear()
        ax.plot(t, noisy, label="Noisy")
        ax.plot(t, clear, label="Clear")
        ax.plot(t, filtered, label="Kalman")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

        # ===== Metrics =====
        m_my = metrics(clear, noisy, filtered)

        text = (
            "KALMAN\n"
            f"MSE  = {m_my['MSE']:.5f}\n"
            f"RMSE = {m_my['RMSE']:.5f}\n"
            f"MAE  = {m_my['MAE']:.5f}\n"
            f"SNR  = {m_my['SNR']:.2f} dB\n"
            f"ΔMSE = {m_my['DELTA_MSE']:.1f} %\n\n"

        )

        self.metrics_label.setText(text)
