import sys
from PyQt6.QtWidgets import QApplication

from filter_params.gui import KalmanWindow


def main():
    app = QApplication(sys.argv)

    window = KalmanWindow()
    window.resize(1100, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
