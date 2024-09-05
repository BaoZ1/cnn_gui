import typing
from PyQt6.QtCore import QSize, pyqtSignal, Qt, QMimeData, QByteArray
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget
from PyQt6.QtGui import QIcon, QMouseEvent
from modelObject import *
from math import ceil

class ObjectList(QListWidget):
  objectClicked = pyqtSignal(ObjectTypes)
  def __init__(self) -> None:
    super().__init__()
    self.setFlow(QListWidget.Flow.LeftToRight)
    self.setWrapping(True)
    self.setFixedWidth(250)

    self.setViewMode(QListWidget.ViewMode.IconMode)
    self.setIconSize(QSize(100, 100))
    self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    self.setDragDropMode(self.DragDropMode.DragOnly)
    self.itemClicked.connect(self.addObject)

    self.itemType = {}
    self.setItems()

  def setItems(self):
    dataSetItem = QListWidgetItem('DataSet')
    dataSetItem.setIcon(QIcon('./icons/dataSet.png'))
    self.itemType[dataSetItem.text()] = ObjectTypes.DataSet

    convItem = QListWidgetItem('Conv')
    convItem.setIcon(QIcon('./icons/conv.png'))
    self.itemType[convItem.text()] = ObjectTypes.Conv

    poolItem = QListWidgetItem('Pool')
    poolItem.setIcon(QIcon('./icons/pool.png'))
    self.itemType[poolItem.text()] = ObjectTypes.Pool

    fullConItem = QListWidgetItem('FullCon')
    fullConItem.setIcon(QIcon('./icons/fc.png'))
    self.itemType[fullConItem.text()] = ObjectTypes.FullConn

    nonlinearItem = QListWidgetItem('Nonlinear')
    nonlinearItem.setIcon(QIcon('./icons/nl.png'))
    self.itemType[nonlinearItem.text()] = ObjectTypes.NonLinear

    classifierItem = QListWidgetItem('Classifier')
    classifierItem.setIcon(QIcon('./icons/classifier.png'))
    self.itemType[classifierItem.text()] = ObjectTypes.Classifier

    errorItem = QListWidgetItem('Error')
    errorItem.setIcon(QIcon('./icons/error.png'))
    self.itemType[errorItem.text()] = ObjectTypes.Error
    

    self.addItem(dataSetItem)
    self.addItem(convItem)
    self.addItem(poolItem)
    self.addItem(fullConItem)
    self.addItem(nonlinearItem)
    self.addItem(classifierItem)
    self.addItem(errorItem)
  
  def addObject(self, *args):
    print(*args)

  def mimeData(self, items: list[QListWidgetItem]) -> QMimeData:
    ret = QMimeData()
    t = self.itemType[items[0].text()]
    ret.setData('objectType', QByteArray.number(t, 10))

    return ret
