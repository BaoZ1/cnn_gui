from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout

class PredictWindow(QWidget):
  def __init__(self) -> None:
    super().__init__()

    self.setWindowTitle('predict result')
    self.predictL = QLabel()
    self.predictL.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.imgL = QLabel()
    ly = QVBoxLayout()
    ly.addWidget(self.predictL)
    ly.addWidget(self.imgL)
    self.setLayout(ly)

  def setContent(self, imgPath: str, label: str):
    self.predictL.setText(label)
    self.imgL.setPixmap(QPixmap(imgPath))
    