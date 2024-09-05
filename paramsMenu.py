from PyQt6.QtCore import pyqtSignal, QSize
from PyQt6.QtWidgets import (QWidget, QLabel, QListWidget, QListWidgetItem, QHBoxLayout,
      QLineEdit, QStackedWidget, QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QPushButton)
from PyQt6.sip import wrappertype
from modelObject import *
import abc

class ParamsMenu(QStackedWidget):
  paramsChanged = pyqtSignal(str, dict)

  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.setFixedWidth(250)
    self.setContentsMargins(0,0,0,0)

    self.emptyMenu = self.addWidget(EmptyParamsMenu())
    self.menus = {
      ObjectTypes.DataSet: self.addWidget(DataSetParamsMenu()),
      ObjectTypes.Conv: self.addWidget(ConvParamsMenu()),
      ObjectTypes.Pool: self.addWidget(PoolParamsMenu()),
      ObjectTypes.FullConn: self.addWidget(FullConnParamsMenu()),
      ObjectTypes.NonLinear: self.addWidget(NonlinearParamsMenu()),
      ObjectTypes.Classifier: self.addWidget(ClassifierParamsMenu()),
      ObjectTypes.Error: self.addWidget(ErrorParamsMenu())
    }
    for id in self.menus.values():
      self.widget(id).paramsChanged.connect(self.updateParams) # type: ignore

    self.currentObj: modelObject | None = None
    

  def showParams(self, obj: modelObject):
    self.currentObj = obj
    self.setCurrentIndex(self.menus[obj.type])
    self.currentWidget().emitSignal = False # type: ignore
    self.currentWidget().setParams(obj.getParams()) # type: ignore
    self.currentWidget().emitSignal = True # type: ignore
  
  def clearParams(self):
    self.currentObj = None
    self.setCurrentIndex(self.emptyMenu)

  def updateParams(self, params):
    if self.currentObj is not None:
      self.currentObj.setPara(params)
  
  def setAllEnabled(self, enabled: bool):
    self.widget(self.emptyMenu).setEnabled(enabled)
    for id in self.menus.values():
      self.widget(id).setEnabled(enabled)

    
class ABListWidgetMeta(wrappertype, abc.ABCMeta):
  ...

class ObjectParamsMenu(QListWidget, abc.ABC, metaclass=ABListWidgetMeta):
  paramsChanged = pyqtSignal(dict)

  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.setFlow(QListWidget.Flow.TopToBottom)
    self.setStyleSheet('border: 0')
    self.setStyleSheet('background-color: 0xdddddd')
    self.setSelectionMode(QListWidget.SelectionMode.NoSelection)

    self.emitSignal = True

  def add(self, name:str, controller: QWidget):
    newI = QListWidgetItem()
    newI.setSizeHint(QSize(0, 50))
    newW = QWidget()
    newW.setMinimumHeight(50)
    newL = QHBoxLayout()
    newL.addWidget(QLabel(name))
    newL.addWidget(controller)
    newW.setLayout(newL)
    self.addItem(newI)
    self.setItemWidget(newI, newW)

  def updateParams(self):
    if self.emitSignal:
      self.paramsChanged.emit(self.getParams())

  @abc.abstractmethod
  def setParams(self, params):
    ...

  @abc.abstractmethod
  def getParams(self) -> dict:
    ...


class EmptyParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.add('Nothing', QWidget())

  def setParams(self, params):
    pass
  
  def getParams(self) -> dict:
    return {}

class DataSetParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.dataPathLE = QLineEdit()
    self.viewBtn = QPushButton('view')
    self.viewBtn.clicked.connect(self.viewDirectories)
    self.dataPathLE.textChanged.connect(self.updateParams)
    pathW = QWidget()
    pathL = QHBoxLayout()
    pathL.addWidget(self.dataPathLE)
    pathL.addWidget(self.viewBtn)
    pathW.setLayout(pathL)
    self.add('data path', pathW)
    
    self.imgHSB = QSpinBox()
    self.imgHSB.setMinimum(1)
    self.imgHSB.valueChanged.connect(self.updateParams)
    self.imgWSB = QSpinBox()
    self.imgWSB.setMinimum(1)
    self.imgWSB.valueChanged.connect(self.updateParams)
    imgSizeW = QWidget()
    imgSizeL = QHBoxLayout()
    imgSizeL.addWidget(self.imgHSB)
    imgSizeL.addWidget(QLabel('x'))
    imgSizeL.addWidget(self.imgWSB)
    imgSizeW.setLayout(imgSizeL)
    self.add('output image size', imgSizeW)

  def viewDirectories(self):
    path = QFileDialog.getExistingDirectory(self, 'select dataset folder')
    self.dataPathLE.setText(path)

  
  def setParams(self, params):
    self.dataPathLE.setText(params['path'])
    self.imgHSB.setValue(params['imgSize'][0])
    self.imgWSB.setValue(params['imgSize'][1])

  def getParams(self) -> dict:
    return {
      'path': self.dataPathLE.text(),
      'imgSize': (self.imgHSB.value(), self.imgWSB.value())
    }
  
class ConvParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.inChannelSB = QSpinBox()
    self.inChannelSB.valueChanged.connect(self.updateParams)
    self.add('input channel count', self.inChannelSB)

    self.outChannelSB = QSpinBox()
    self.outChannelSB.valueChanged.connect(self.updateParams)
    self.add('output channel count', self.outChannelSB)

    self.kernelHSB = QSpinBox()
    self.kernelHSB.setMinimum(1)
    self.kernelHSB.valueChanged.connect(self.updateParams)
    self.kernelWSB = QSpinBox()
    self.kernelWSB.setMinimum(1)
    self.kernelWSB.valueChanged.connect(self.updateParams)
    kernelSizeW = QWidget()
    kernelSizeL = QHBoxLayout()
    kernelSizeL.addWidget(self.kernelHSB)
    kernelSizeL.addWidget(QLabel('x'))
    kernelSizeL.addWidget(self.kernelWSB)
    kernelSizeW.setLayout(kernelSizeL)
    self.add('kernel size', kernelSizeW)

    self.strideSB = QSpinBox()
    self.strideSB.setMinimum(1)
    self.strideSB.setEnabled(False)
    self.strideSB.valueChanged.connect(self.updateParams)
    self.add('stride', self.strideSB)

  
  def setParams(self, params):
    self.inChannelSB.setValue(params['inChannel'])
    self.outChannelSB.setValue(params['outChannel'])
    self.kernelHSB.setValue(params['kernelSize'][0])
    self.kernelWSB.setValue(params['kernelSize'][1])
    self.strideSB.setValue(params['stride'])

  def getParams(self) -> dict:
    return {
      'inChannel': self.inChannelSB.value(),
      'outChannel': self.outChannelSB.value(),
      'kernelSize': (self.kernelHSB.value(), self.kernelWSB.value()),
      'stride': self.strideSB.value()
    }
  
class PoolParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.sizeSB = QSpinBox()
    self.sizeSB.setMinimum(1)
    self.sizeSB.valueChanged.connect(self.updateParams)
    self.add('size', self.sizeSB)

    self.strideSB = QSpinBox()
    self.strideSB.setMinimum(1)
    self.strideSB.valueChanged.connect(self.updateParams)
    self.add('stride', self.strideSB)

    self.typeCB = QComboBox()
    self.typeCB.addItems(('max', 'min', 'average'))
    self.typeCB.currentTextChanged.connect(self.updateParams)
    self.add('type', self.typeCB)

  
  def setParams(self, params):
    self.sizeSB.setValue(params['size'])
    self.strideSB.setValue(params['stride'])
    self.typeCB.setCurrentText(params['type'])

  def getParams(self) -> dict:
    return {
      'size': self.sizeSB.value(),
      'type': self.typeCB.currentText(),
      'stride': self.strideSB.value()
    }
  
class FullConnParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.inSizeSB = QSpinBox()
    self.inSizeSB.setRange(1, 99999)
    self.inSizeSB.valueChanged.connect(self.updateParams)
    self.add('input size', self.inSizeSB)

    self.outSizeSB = QSpinBox()
    self.outSizeSB.setRange(1, 99999)
    self.outSizeSB.valueChanged.connect(self.updateParams)
    self.add('output size', self.outSizeSB)

  
  def setParams(self, params):
    self.inSizeSB.setValue(params['inSize'])
    self.outSizeSB.setValue(params['outSize'])

  def getParams(self) -> dict:
    return {
      'inSize': self.inSizeSB.value(),
      'outSize': self.outSizeSB.value()
    }
  
class NonlinearParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.typeCB = QComboBox()
    self.typeCB.addItems(('relu', 'sigmoid', 'tanh'))
    self.typeCB.editTextChanged.connect(self.updateParams)
    self.add('activation function', self.typeCB)

  
  def setParams(self, params):
    self.typeCB.setCurrentText(params['func'])

  def getParams(self) -> dict:
    return {
      'func': self.typeCB.currentText()
    }
  
class ClassifierParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.inSizeSB = QSpinBox()
    self.inSizeSB.setRange(1, 99999)
    self.inSizeSB.valueChanged.connect(self.updateParams)
    self.add('input size', self.inSizeSB)

    self.classCountSB = QSpinBox()
    self.classCountSB.setRange(1, 99999)
    self.classCountSB.valueChanged.connect(self.updateParams)
    self.add('class count', self.classCountSB)

  
  def setParams(self, params):
    self.inSizeSB.setValue(params['inSize'])
    self.classCountSB.setValue(params['classCount'])

  def getParams(self) -> dict:
    return {
      'inSize': self.inSizeSB.value(),
      'classCount': self.classCountSB.value()
    }

class ErrorParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.funcCB = QComboBox()
    self.funcCB.addItems(('ce', 'mae', 'mse'))
    self.funcCB.editTextChanged.connect(self.updateParams)
    self.add('loss function', self.funcCB)

  
  def setParams(self, params):
    self.funcCB.setCurrentText(params['func'])

  def getParams(self) -> dict:
    return {
      'func': self.funcCB.currentText()
    }
  

class HyperParamsMenu(ObjectParamsMenu):
  def __init__(self, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self.setFixedWidth(200)

    self.batchSizeSB = QSpinBox()
    self.batchSizeSB.setMinimum(1)
    self.batchSizeSB.valueChanged.connect(self.updateParams)
    self.add('batch size', self.batchSizeSB)

    self.lrSB = QDoubleSpinBox()
    self.lrSB.setRange(0, 1)
    self.lrSB.setDecimals(6)
    self.lrSB.valueChanged.connect(self.updateParams)
    self.add('learning rate', self.lrSB)
    
    self.icSB = QSpinBox()
    self.icSB.setRange(1, 1000)
    self.icSB.valueChanged.connect(self.updateParams)
    self.add('iteration count', self.icSB)
  
  def setParams(self, params):
    self.batchSizeSB.setValue(params['batchSize'])
    self.lrSB.setValue(params['learningRate'])
    self.icSB.setValue(params['iterCount'])

  def getParams(self) -> dict:
    return {
      'batchSize': self.batchSizeSB.value(),
      'learningRate': self.lrSB.value(),
      'iterCount': self.icSB.value()
    }