import typing
from PyQt6.QtCore import QPointF, Qt, pyqtSignal, QRect, QMimeData, QThread
from PyQt6.QtGui import (QPainter, QPen, QPixmap, QDrag,
  QDragEnterEvent, QDropEvent, QPaintEvent, QMouseEvent, QWheelEvent, QDragMoveEvent)
from PyQt6.QtWidgets import QWidget, QListWidget, QFileDialog
from modelObject import *
from console import LogType
from predictWindow import PredictWindow
import pickle

transable = typing.TypeVar('transable', QPointF, int, float)

class EditingArea(QWidget):
  editObject = pyqtSignal(modelObject)
  clearParams = pyqtSignal()
  log = pyqtSignal(str, LogType)
  trainFinished = pyqtSignal()
  updateHyperParamsDisplay = pyqtSignal(dict)

  objectClasses = {
    ObjectTypes.DataSet: DataSet,
    ObjectTypes.Conv: Conv,
    ObjectTypes.Pool: Pool,
    ObjectTypes.FullConn: FullConn,
    ObjectTypes.NonLinear: NonLinear,
    ObjectTypes.Classifier: Classifier,
    ObjectTypes.Error: Error
  }
  imgs = {
    ObjectTypes.DataSet: 'icons/dataSet.png',
    ObjectTypes.Conv: 'icons/conv.png',
    ObjectTypes.Pool: 'icons/pool.png',
    ObjectTypes.FullConn: 'icons/fc.png',
    ObjectTypes.NonLinear: 'icons/nl.png',
    ObjectTypes.Classifier: 'icons/classifier.png',
    ObjectTypes.Error: 'icons/error.png'
  }

  def __init__(self, parent = None) -> None:
    super().__init__(parent)
    self.setAcceptDrops(True)
    self.setMouseTracking(True)
    self.setMinimumSize(350, 300)

    self.model = AllModelObj()

    self.objSize = lambda: (int(100 * self.scale), int(100 * self.scale))

    self.origin = QPointF(0, 0)
    self.scale = 1
    self.lastPos = QPointF()

    self.draggingObj: modelObject | None = None

    self.connectBeginObj: modelObject | None = None
    self.connectionPos = None

    self.trainThread: TrainThread | None = None

    self.predictW = PredictWindow()


  def paintEvent(self, a0: QPaintEvent) -> None:
    pt = QPainter()
    pt.begin(self)
    pt.setPen(0xffffff)
    pt.setBrush(0xffffff)
    pt.drawRect(self.rect())

    pt.setPen(QPen(0x0, 3, Qt.PenStyle.SolidLine))
    for conn in self.model.connections.connections.values():
      beginP = self.trans(QPointF(*conn.NobjS.pos))
      endP = self.trans(QPointF(*conn.NobjE.pos))
      pt.drawLine(beginP, endP)
      direction = endP - beginP # type: ignore
      direction *= (direction.x() ** 2 + direction.y() ** 2) ** -0.5
      rotDirection = QPointF(-direction.y(), direction.x())
      middleP1 = (beginP + endP) * 0.5 - direction * 5 + rotDirection * 5 # type: ignore
      middleP2 = (beginP + endP) * 0.5 - direction * 5 - rotDirection * 5 # type: ignore
      middleP3 = (beginP + endP) * 0.5 + direction * 5 # type: ignore
      pt.drawLine(middleP1, middleP3)
      pt.drawLine(middleP2, middleP3)

    if self.connectionPos is not None and self.connectBeginObj is not None:
      pt.setPen(QPen(0x0, 3, Qt.PenStyle.DashDotLine))
      pt.drawLine(self.trans(QPointF(*self.connectBeginObj.pos)), self.connectionPos)

    for id in self.model.objects:
      obj = self.model.objects[id]
      center = self.trans(QPointF(*obj.pos))
      size = QPointF(*self.objSize())
      pt.drawPixmap(
        QRect((center - size * 0.5).toPoint(), (center + size * 0.5).toPoint()), # type: ignore
        QPixmap(self.imgs[obj.type]))
    pt.end()

  def trans(self, p: QPointF) -> QPointF:
    return (p - self.origin) * self.scale # type: ignore
  
  def retrans(self, p: QPointF) -> QPointF:
    return (p / self.scale) + self.origin # type: ignore
    
  def mousePressEvent(self, a0: QMouseEvent) -> None:
    currentObj = self.getCurrentObj(a0.position())
    if a0.button() == Qt.MouseButton.LeftButton:
      self.lastPos = a0.position()
      if currentObj is not None:
        self.editObject.emit(currentObj)
        self.model.objects.move_to_end(currentObj.id)
        self.draggingObj = currentObj
        self.repaint()
      else:
        self.clearParams.emit()
    elif a0.button() == Qt.MouseButton.RightButton:
      if currentObj is not None:
        drag = QDrag(self)
        mimeData = QMimeData()
        mimeData.setText(currentObj.id)
        drag.setMimeData(mimeData)
        self.connectBeginObj = currentObj
        drag.exec(Qt.DropAction.LinkAction)
        self.connectBeginObj = None
        self.repaint()
    return super().mousePressEvent(a0)
  
  def mouseMoveEvent(self, a0: QMouseEvent) -> None:
    if a0.buttons() & Qt.MouseButton.LeftButton:
      delta = (a0.position() - self.lastPos) / self.scale # type: ignore
      #currentObj = self.getCurrentObj(a0.position())
      if self.draggingObj is None:
        self.origin -= delta
      else:
        self.draggingObj.pos = (self.draggingObj.pos[0] + delta.x(), self.draggingObj.pos[1] + delta.y())
      self.lastPos = a0.position()
      self.repaint()
    else:
      self.draggingObj = None
    return super().mouseMoveEvent(a0)

  def getCurrentObj(self, position):
    for id in reversed(self.model.objects.keys()):
      obj = self.model.objects[id]
      center = self.trans(QPointF(*obj.pos))
      if abs(position.x() - center.x()) < self.objSize()[0] / 2 \
          and abs(position.y() - center.y()) < self.objSize()[1] / 2:
        return obj
    return None
  
  def wheelEvent(self, a0: QWheelEvent) -> None:
    constPos = self.retrans(a0.position()) # type: ignore
    if a0.angleDelta().y() > 0:
      self.scale *= 1.1
      self.origin = constPos - a0.position() / self.scale # type: ignore
      self.repaint()
    elif a0.angleDelta().y() < 0:
      self.scale /= 1.1
      self.origin = constPos - a0.position() / self.scale # type: ignore
      self.repaint()

  def dragEnterEvent(self, a0: QDragEnterEvent) -> None:
    a0.acceptProposedAction()

  def dragMoveEvent(self, a0: QDragMoveEvent) -> None:
    if isinstance(a0.source(), EditingArea):
      self.connectionPos = a0.position()
      self.repaint()

  def dropEvent(self, a0: QDropEvent) -> None:
    if self.trainThread is not None:
      self.log.emit('cannot edit model while training', LogType.warning)
    if isinstance(a0.source(), QListWidget):
      t = ObjectTypes(int(a0.mimeData().data('objectType').data()))
      p = self.retrans(a0.position())
      newObj = self.objectClasses[t](p.x(), p.y())
      try:
        self.model.add(newObj)
      except Exception as e:
        self.log.emit(str(e), LogType.error)
      else:
        self.editObject.emit(newObj)
        self.log.emit(f'added new object: {newObj.id}', LogType.normal)
    elif isinstance(a0.source(), EditingArea):
      endObj = self.getCurrentObj(a0.position())
      if self.connectBeginObj is not None and endObj is not None and endObj != self.connectBeginObj:
        try:
          self.model.connections.add(Connection(self.connectBeginObj, endObj))
        except Exception as e:
          self.log.emit(str(e), LogType.error)
        else:
          self.log.emit(f'connected from {self.connectBeginObj.id} to {endObj.id}', LogType.normal)
        
    self.repaint()

  def clear(self):
    self.model = AllModelObj()
    self.updateHyperParamsDisplay.emit(self.model.getParams())
    print('clear')
    self.repaint()
  
  def save(self):
    path, _ = QFileDialog.getSaveFileName(self, 'save model', filter='model files(*.model)')
    if os.path.isfile(path):
      with open(path, 'wb') as f:
        pickle.dump(self.model, f)

  def load(self):
    path, _ = QFileDialog.getOpenFileName(self, 'select model file', filter='model files(*.model)')
    if os.path.isfile(path):
      with open(path, 'rb') as f:
        self.model = pickle.load(f)
        self.updateHyperParamsDisplay.emit(self.model.getParams())
      self.repaint()

  def check(self):
    errConns = self.model.connections.check()
    if len(errConns) == 0:
      self.log.emit('all connections are correct', LogType.normal)
    else:
      self.log.emit(f'found {len(errConns)} wrong connections:', LogType.error)
      for errConn in errConns:
        self.log.emit(
          f'{errConn.NobjS.id} -> {errConn.NobjE.id}: cannot connect {errConn.NobjS.type.name} to {errConn.NobjE.type.name}',
           LogType.error)
        
  def setHyperParams(self, params: dict):
    self.model.setPara(params)

  def initModel(self):
    try:
      self.model.init()
    except Exception as e:
      self.log.emit(str(e), LogType.error)
    else:
      self.log.emit('all object initialization completed', LogType.normal)

  def train(self):
    self.log.emit('==========training==========', LogType.normal)
    self.trainThread = TrainThread(self.model)
    self.trainThread.epochFinished.connect(
      lambda epoch, res: self.log.emit(
      f'{epoch+1}--- train acc:{round(res[0], 4)}; test acc:{round(res[1], 4)}; average loss:{round(res[2], 4)}', LogType.normal)
    )
    self.trainThread.finished.connect(self.finishTrain)
    self.trainThread.error.connect(
      lambda msg: self.log.emit(msg, LogType.error)
    )
    self.trainThread.start()
  
  def finishTrain(self):
    self.log.emit('finished', LogType.normal)
    self.trainThread = None
    self.trainFinished.emit()
  
  def interruptTrain(self):
    if self.trainThread is not None:
      self.trainThread.requestInterruption()
      self.log.emit('interrupting', LogType.warning)
  
  def predict(self):
    if self.trainThread is not None:
      self.log.emit('cannot predict while model is training', LogType.warning)
      return

    path, _ = QFileDialog.getOpenFileName(self, 'select image', filter='image file(*.jpg *.jpeg *.png)')
    if os.path.isfile(path):
      img = Image.open(path)
      pred = self.model.predict(img)
      print(pred)
      self.predictW.setContent(path, pred)
      self.predictW.show()
      


class TrainThread(QThread):
  epochFinished = pyqtSignal(int, tuple)
  error = pyqtSignal(str)

  def __init__(self, model: AllModelObj) -> None:
    super().__init__()
    self.model = model

  def run(self):
    try:
      for i in range(self.model.iterCount):
        self.epochFinished.emit(i, self.model.procFunc())
        if self.isInterruptionRequested():
          break
    except Exception as e:
      self.error.emit(str(e))

  def requestInterruption(self) -> None:
    self.model.interrupt()
    return super().requestInterruption()