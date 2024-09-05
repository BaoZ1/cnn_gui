from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QPushButton, QListWidget, QListWidgetItem, QVBoxLayout
from enum import IntEnum

class LogType(IntEnum):
  normal = 1
  warning = 2
  error = 3

class Console(QWidget):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)

    self.lst = QListWidget()
    self.clearBtn = QPushButton('clear')
    self.clearBtn.clicked.connect(self.lst.clear)

    vl = QVBoxLayout()
    vl.setContentsMargins(0, 0, 0, 0)
    vl.addWidget(self.lst)
    vl.addWidget(self.clearBtn)
    vl.setAlignment(self.clearBtn, Qt.AlignmentFlag.AlignRight)
    self.setLayout(vl)
  
  def add(self, text, type):
    newI = QListWidgetItem(text)
    match type:
      case LogType.normal:
        newI.setForeground(0x0)
      case LogType.warning:
        newI.setForeground(0xe89817)
      case LogType.error:
        newI.setForeground(0xff0000)
    self.lst.addItem(newI)

 