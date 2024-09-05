import sys
from PyQt6.QtCore import QFile
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from objectList import ObjectList
from editingArea import EditingArea
from paramsMenu import ParamsMenu, HyperParamsMenu
from console import Console

class MainWidget(QWidget):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle('CNN Creator')

    self.setupUi()

  def setupUi(self):
    self.loadModelBtn = QPushButton('load model')
    self.createModelBtn = QPushButton('create new model')
    self.setParamsBtn = QPushButton('set parameters')
    self.saveModelBtn = QPushButton('save model')
    self.checkModelBtn = QPushButton('check model')

    self.toolBarL = QHBoxLayout()
    self.toolBarL.setObjectName('toolBarL')
    self.toolBarL.addWidget(self.loadModelBtn)
    self.toolBarL.addWidget(self.createModelBtn)
    self.toolBarL.addWidget(self.setParamsBtn)
    self.toolBarL.addWidget(self.saveModelBtn)
    self.toolBarL.addWidget(self.checkModelBtn)
    self.toolBarL.addStretch()

    self.objectList = ObjectList()
    self.objectList.setObjectName('objectList')
    self.editingArea = EditingArea()
    self.paramsMenu = ParamsMenu()

    self.middleL = QHBoxLayout()
    self.middleL.addWidget(self.objectList)
    self.middleL.addWidget(self.editingArea)
    self.middleL.addWidget(self.paramsMenu)

    self.modelInitBtn = QPushButton('init model')
    self.trainBtn = QPushButton('train')
    self.interruptBtn = QPushButton('interrupt')
    self.predictBtn = QPushButton('predict')
    self.interruptBtn.setVisible(False)

    self.middleBtnsL = QHBoxLayout()
    self.middleBtnsL.addWidget(self.modelInitBtn)
    self.middleBtnsL.addWidget(self.trainBtn)
    self.middleBtnsL.addWidget(self.interruptBtn)
    self.middleBtnsL.addWidget(self.predictBtn)
    self.middleBtnsL.addStretch()

    self.console = Console()
    self.hpMenu = HyperParamsMenu()
    
    self.bottomL = QHBoxLayout()
    self.bottomL.addWidget(self.console)
    self.bottomL.addWidget(self.hpMenu)

    self.mainL = QVBoxLayout()
    self.mainL.setContentsMargins(20, 20, 20, 20)
    self.mainL.addLayout(self.toolBarL)
    self.mainL.addLayout(self.middleL)
    self.mainL.addLayout(self.middleBtnsL)
    self.mainL.addLayout(self.bottomL)
    self.setLayout(self.mainL)

    self.loadModelBtn.clicked.connect(self.editingArea.load)
    self.createModelBtn.clicked.connect(self.editingArea.clear)
    #self.setParamsBtn.clicked.connect(self.setParams)
    self.saveModelBtn.clicked.connect(self.editingArea.save)
    self.checkModelBtn.clicked.connect(self.editingArea.check)

    self.editingArea.editObject.connect(self.paramsMenu.showParams)
    self.editingArea.clearParams.connect(self.paramsMenu.clearParams)
    self.editingArea.trainFinished.connect(self.trainStoped)

    self.hpMenu.paramsChanged.connect(self.editingArea.setHyperParams)

    self.modelInitBtn.clicked.connect(self.editingArea.initModel)
    self.trainBtn.clicked.connect(self.train)
    self.interruptBtn.clicked.connect(self.interruptTrain)
    self.predictBtn.clicked.connect(self.editingArea.predict)

    self.editingArea.log.connect(self.console.add)
    self.editingArea.updateHyperParamsDisplay.connect(self.hpMenu.setParams)
    self.editingArea.updateHyperParamsDisplay.emit(self.editingArea.model.getParams())
  
  def train(self):
    self.editingArea.train()
    self.modelInitBtn.setEnabled(False)
    self.trainBtn.setVisible(False)
    self.interruptBtn.setVisible(True)

    self.paramsMenu.setAllEnabled(False)
    self.hpMenu.setEnabled(False)

  def trainStoped(self):
    self.modelInitBtn.setEnabled(True)
    self.trainBtn.setVisible(True)
    self.interruptBtn.setVisible(False)

    self.paramsMenu.setAllEnabled(True)
    self.hpMenu.setEnabled(True)
  
  def interruptTrain(self):
    self.editingArea.interruptTrain()
    self.modelInitBtn.setEnabled(True)
    self.trainBtn.setVisible(True)
    self.interruptBtn.setVisible(False)

    self.paramsMenu.setAllEnabled(True)
    self.hpMenu.setEnabled(True)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  qssFile = QFile('style.qss')
  qssFile.open(QFile.OpenModeFlag.ReadOnly)
  app.setStyleSheet(str(qssFile.readAll().data(), encoding='utf-8'))
  qssFile.close()
  w = MainWidget()
  w.show()
  sys.exit(app.exec())
