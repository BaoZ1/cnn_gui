from abc import ABC, abstractmethod
from enum import IntEnum
import typing
import numpy as np
import os
from PIL import Image
from collections import OrderedDict
import random


class ObjectTypes(IntEnum):

  DataSet = 1
  Conv = 2
  Pool = 3
  FullConn = 4
  NonLinear = 5
  Classifier = 6
  Error = 7
  AjConv = 8
  AjFullConn = 9


def reshape(data: np.ndarray, dimCount: int):
  missingCount = dimCount - len(data.shape)
  if missingCount < 0:
    raise Exception('wrong shape')
  return np.expand_dims(data, axis=[*range(missingCount)])


class modelObject(ABC):
  def __new__(cls, *args, **kwargs):
    if 'count' not in cls.__dict__.keys():
      cls.count = 1
    else:
      cls.count += 1
    return super().__new__(cls)
  
  def __init__(self, x = 0, y = 0):
    self.id = self.__class__.__name__ + f'{self.__class__.count}'
    self.label = self.id
    self.pos = (x, y)
    self.inited = False

  def __repr__(self):
    s = ', '.join([
      self.id,
      str(self.type.value),
      f'\"{self.label}\"',
      self.procFunc.__name__,
      self.setParaFunc.__name__,
      self.paraString(),
      str(self.pos)
    ])
    return f'<{s}>'

  @abstractmethod
  def paraString(self) -> str:
    ...

  def _init(self, type: ObjectTypes, procFunc: typing.Callable, adjustFunc: typing.Callable, setParaFunc: typing.Callable):
    self.type = type
    self.procFunc = procFunc
    self.ajFunc = adjustFunc
    self.setParaFunc = setParaFunc
    self.setDefaultPara()
    self.inited = False

  def process(self, *data): 
    assert self.inited, f'{self.id} has not been initialized'
    return self.procFunc(*data)
  
  def adjust(self, errors, lr):
    return self.ajFunc(errors, lr)

  def setPara(self, para: dict):
    if self.getParams() != para:
      self.setParaFunc(para)
      self.inited = False

  @abstractmethod
  def getParams(self) -> dict:
    ...

  @abstractmethod
  def setDefaultPara(self):
    ...

  @abstractmethod
  def init(self):
    ...


class DataSet(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.DataSet,
      self.loadData,
      self.ajDataSetProc,
      self.setDataPara
    )

  def paraString(self):
    return self.imgFolder
  
  def setDefaultPara(self):
    self.imgFolder = './'
    self.imgSize = (32, 32)

  def init(self):
    try:
      folders = next(os.walk(self.imgFolder))[1]
    except Exception: 
      raise Exception('invalid dataset path')
    self.labels = folders
    dataList = []
    m = np.identity(len(folders))
    for index, folder in enumerate(folders):
      folderPath = os.path.join(self.imgFolder, folder)
      for imgName in next(os.walk(folderPath))[2]:
        dataList.append((os.path.join(folderPath, imgName), m[index]))
    random.shuffle(dataList)
    testCount = int(len(dataList) * 0.2)
    self.testList =dataList[:testCount]
    self.trainList = dataList[testCount:]
    self.inited = True

  def loadData(self, batchSize):
    counter = 0

    while len(self.trainList) - counter >= batchSize:
      imgs = []
      labels = []
      for _ in range(batchSize):
        counter += 1
        img = Image.open(self.trainList[counter][0])
        """ if img.layers == 1:
          print('1111') """
        img = img.resize(self.imgSize)
        img = np.asarray(img)
        img = img.transpose((2,0,1))
        """ if img.shape != (3, *self.imgSize):
          print('222')
          continue """
        img = (img - img.mean()) / img.std()
        imgs.append(img)
        labels.append(self.trainList[counter][1])
      imgs = np.array(imgs)
      labels = np.array(labels)
      yield imgs, labels
    
  def testData(self):
    imgs = []
    labels = []
    for imgPath, onehotLabel in self.testList:
      img = Image.open(imgPath)
      img = img.resize(self.imgSize)
      img = np.asarray(img)
      img = img.transpose((2,0,1))
      if img.shape != (3, *self.imgSize):
        continue
      img = (img - img.mean()) / img.std()
      imgs.append(img)
      labels.append(onehotLabel)
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

  def ajDataSetProc(self, *args):
    pass

  def getParams(self) -> dict:
    return {
      'path': self.imgFolder,
      'imgSize': self.imgSize
    }

  def setDataPara(self, params: dict | None = None):
    if params is None:
      self.imgFolder = '/'
      self.imgSize = (32, 32)
    else:
      self.imgFolder = params['path']
      self.imgSize = params['imgSize']


class Conv(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.Conv,
      self.convProc,
      self.ajConvProc,
      self.setConvPara
    )

  def paraString(self):
    content = ', '.join([
      str(self.kernelSize),
      str(self.stride)
    ])
    return f'[{content}]'
  
  def setDefaultPara(self):
    self.kernelSize = (3, 3)
    self.channelCount = (3, 3)
    self.stride = 1
  
  def convProc(self, data: np.ndarray):
    data = reshape(data, 4)
    self.inputs = data

    assert data.shape[1] == self.channelCount[0], f'Wrong number of input channels: input {data.shape[1]}'
    featureMapSize = (
      (data.shape[2] - self.kernelSize[0]) // self.stride + 1,
      (data.shape[3] - self.kernelSize[1]) // self.stride + 1
    )
    out = np.zeros((data.shape[0], self.channelCount[1], *featureMapSize))
    for i in range(data.shape[0]):
      out[i] = self.convImage(data[i], featureMapSize)
    
    return out
  
  def ajConvProc(self, errors: np.ndarray, lr):
    tmpData = np.zeros((
      *errors.shape[:2],
      errors.shape[2] + 2 * (self.kernelSize[0] - 1),
      errors.shape[3] + 2 * (self.kernelSize[1] - 1)
    ))
    tmpData[:, :, self.kernelSize[0] - 1 : -(self.kernelSize[0] - 1),
         self.kernelSize[0] - 1 : -(self.kernelSize[0] - 1)] = errors
    tmpData = np.expand_dims(tmpData, axis=1)
    tmpKernel = np.rot90(self.kernels, 2, axes=(2, 3)).transpose((1,0,2,3))
    ret = np.zeros_like(self.inputs, dtype=np.float64)
    """ for sample in range(errors.shape[0]):
      for inChannel in range(self.channelCount[0]):
        for outChannel in range(self.channelCount[1]):
          ret[sample, inChannel] += \
            self.conv(tmpData[sample, inChannel], tmpKernel[inChannel, outChannel], ret.shape[2:]) """
    # tmpData.shape = (batchSize, 1, outChannel, ..., ...) 
    # tmpKernel.shape = (inChannel, outChannel, ..., ...)
    # ret.shape = (batchSize, inChannel, ..., ...)
    #
    # (convArea * tmpData).shape = (batchSize, inChannel, outChannel, ..., ...)
    for i in range(ret.shape[2]):
      for j in range(ret.shape[3]):
        lt = (i * self.stride, j * self.stride)
        convArea = tmpData[:, :, :, lt[0]:lt[0]+tmpKernel.shape[2], lt[1]:lt[1]+tmpKernel.shape[3]]
        ret[:, :, i, j] = np.sum(convArea * tmpKernel, axis=(2,3,4))

    # errors.shape = (batchSize, outChannel, .., ...)
    #             -> (outChannel, 1, batchSize, ..., ...)
    # inputs.shape = (batchSize, inChannel, ..., ...)
    #             -> (inChannel, batchSize, ..., ...)
    #
    # (input * errors).shape = (outChannel, inChannel, batchSize, ..., ...)
    errors = np.expand_dims(errors.transpose(1,0,2,3), axis=1)
    self.inputs = self.inputs.transpose((1,0,2,3))
    kd = np.zeros_like(self.kernels)
    for i in range(self.kernelSize[0]):
      for j in range(self.kernelSize[1]):
        lt = (i * self.stride, j * self.stride)
        convArea = self.inputs[:, :, lt[0]:lt[0]+errors.shape[3], lt[1]:lt[1]+errors.shape[4]]
        kd[:, :, i, j] = np.sum(convArea * errors, axis=(2,3,4))
    self.kernels -= kd * lr
    """ for inChannel in range(self.channelCount[0]):
      for outChannel in range(self.channelCount[1]):
        kd = np.zeros(self.kernelSize)
        for sample in range(errors.shape[0]):
          kd += self.conv(self.inputs[sample, inChannel], errors[sample, outChannel], self.kernelSize)
        self.kernels[inChannel, outChannel] -= kd * lr """
    self.bias -= errors.sum((1,2,3,4)) * lr

    return ret

  def init(self):
    self.kernels = np.random.random((*self.channelCount[::-1], *self.kernelSize)) * 0.1 - 0.05
    self.bias = np.random.random(self.channelCount[1]) * 0.1 - 0.05
    self.inited = True

  def convImage(self, img: np.ndarray, featureMapSize: typing.Tuple[int, int]):
    featureMaps = np.zeros((self.channelCount[1], *featureMapSize))

    for i in range(featureMapSize[0]):
      for j in range(featureMapSize[1]):
        lt = (i * self.stride, j * self.stride)
        convArea = img[:, lt[0]:lt[0]+self.kernelSize[0], lt[1]:lt[1]+self.kernelSize[1]]
        featureMaps[:, i, j] = np.sum(convArea * self.kernels, axis=(1,2,3))
    """ for inChannel, kernels in enumerate(self.kernels):
      for outChannel, kernel in enumerate(kernels):
        featureMaps[outChannel] += self.conv(img[inChannel], kernel, featureMapSize) + self.bias[outChannel]
 """
    return featureMaps

  """ def conv(self, data: np.ndarray, kernel: np.ndarray, featureMapSize: typing.Tuple[int, int]):
    featureMap = np.zeros(featureMapSize)
    for i in range(featureMapSize[0]):
      for j in range(featureMapSize[1]):
        lt = (i * self.stride, j * self.stride)
        convArea = data[:, lt[0]:lt[0]+kernel.shape[0], lt[1]:lt[1]+kernel.shape[1]]
        featureMap[i][j] = np.sum(convArea * kernel)

    return featureMap """
  
  def getParams(self) -> dict:
    return {
      'inChannel': self.channelCount[0],
      'outChannel': self.channelCount[1],
      'kernelSize': self.kernelSize,
      'stride': self.stride
    }
  
  def setConvPara(self, params: dict):
    self.channelCount = (params['inChannel'], params['outChannel'])
    self.kernelSize = params['kernelSize']
    self.stride = params['stride']

class Pool(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.Pool,
      self.poolProc,
      self.ajPoolProc,
      self.setPoolPara
    )

  def paraString(self):
    return 'Pool parameters'
  
  def setDefaultPara(self):
    self.size = 2
    self.poolType = 'max'
    self.stride = 2
  
  def poolProc(self, data: np.ndarray):
    data = reshape(data, 4)
    self.inputSize = data.shape
    out = np.zeros((
      *data.shape[:2],
      (data.shape[2] - self.size) // self.stride + 1,
      (data.shape[3] - self.size) // self.stride + 1,
    ))
    self.de = np.zeros_like(data, dtype=np.float64)

    for m, sample in enumerate(data):
      for n, channel in enumerate(sample):
        for i in range(out.shape[2]):
          for j in range(out.shape[3]):
            lt = (i * self.stride, j * self.stride)
            area = channel[lt[0]:lt[0]+self.size, lt[1]:lt[1]+self.size]
            v = self.func(area)
            out[m][n][i][j] = v
            self.de[m][n][lt[0]:lt[0]+self.size, lt[1]:lt[1]+self.size] += area == v
    #self.de /= self.de.sum((1,2,3)).reshape(-1, 1, 1, 1)
    #print(self.de)
    return out
  
  def ajPoolProc(self, errors: np.ndarray, lr):
    ret = np.zeros(self.inputSize)
    for m, sample in enumerate(ret):
      for n, channel in enumerate(sample):
        for i in range(errors.shape[2]):
          for j in range(errors.shape[3]):
            lt = (i * self.stride, j * self.stride)
            if self.func in (self.max, self.min):
              e = errors[m, n, i, j] * self.de[m, n, i, j]
            elif self.func == self.average:
              e = errors[m, n, i, j] * (1 / np.prod(self.de.shape[1:]))
            else:
              raise Exception("???")
            channel[lt[0]:lt[0]+self.size, lt[1]:lt[1]+self.size] += e
    return ret


  def init(self):
    match self.poolType:
      case 'max':
        self.func = self.max
      case 'min':
        self.func = self.min
      case 'average':
        self.func = self.average
      case _:
        raise Exception('wrong pool func type')
    self.inited = True

  def max(self, *args):
    return np.max(*args)

  def min(self, *args):
    return np.min(*args)
  
  def average(self, *args):
    return np.mean(*args)
  
  def getParams(self) -> dict:
    return {
      'size': self.size,
      'type': self.poolType,
      'stride': self.stride
    }

  def setPoolPara(self, params: dict):
    self.size = params['size']
    self.poolType = params['type']
    self.stride = params['stride']

class FullConn(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.FullConn,
      self.fullConnProc,
      self.ajFullConnProc,
      self.setFullConnPara
    )

  def paraString(self):
    return f'{self.shape}'
  
  def setDefaultPara(self):
    self.shape = (5, 5)
  
  def fullConnProc(self, data: np.ndarray):
    self.inputShape = data.shape
    data = data.reshape(-1, self.shape[0])
    self.de = np.expand_dims(data, 2)
    return data @ self.w + self.b
  
  def ajFullConnProc(self, errors: np.ndarray, lr):
    self.w -= (self.de @ np.expand_dims(errors, 1)).mean(0) * lr
    self.b -= errors.mean(0) * lr
    return (errors @ self.w.T).reshape(self.inputShape)
  
  def init(self):
    self.w = np.random.random(self.shape) * 0.1 - 0.05
    self.b = np.random.random(self.shape[1]) * 0.1 - 0.05
    self.inited = True
    
  def getParams(self) -> dict:
    return {
      'inSize': self.shape[0],
      'outSize': self.shape[1]
    }

  def setFullConnPara(self, params: dict):
    self.shape = (params['inSize'], params['outSize'])

class NonLinear(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.NonLinear,
      self.nonLinearProc,
      self.ajNonLinearProc,
      self.setNonLinearPara
    )

  def paraString(self):
    return self.funcPara
  
  def setDefaultPara(self):
    self.funcPara = 'sigmoid'
    
  def init(self):
    match self.funcPara:
      case 'relu':
        self.func = self.relu
      case 'sigmoid':
        self.func = self.sigmoid
      case 'tanh':
        self.func = self.tanh
      case _:
        raise Exception('wrong nonlinear type')
    self.inited = True
  
  def nonLinearProc(self, data):
    return self.func(data)
  
  def ajNonLinearProc(self, errors, lr):
    return self.de * errors

  def relu(self, data: np.ndarray):
    self.de = (data > 0).astype(np.int32)
    return np.maximum(data, 0)
  
  def sigmoid(self, data):
    s = 1 / (1 + np.exp(-data))
    self.de = s * (1-s)
    return s
  
  def tanh(self, data):
    s = np.tanh(data)
    self.de = 1 - s * s
    return s

  def getParams(self) -> dict:
    return {
      'func': self.funcPara
    }
  
  def setNonLinearPara(self, params: dict):
    self.funcPara = params['func']

class Classifier(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self.fc = FullConn()
    self._init(
      ObjectTypes.Classifier,
      self.classifierProc,
      self.ajClassifierProc,
      self.setClassifierPara
    )

  def paraString(self):
    return self.fc.paraString()
  
  def setDefaultPara(self):
    self.fc.setDefaultPara()
  
  def classifierProc(self, data):
    res = self.fc.process(data)
    return res
  
  def ajClassifierProc(self, errors, lr):
    ret = self.fc.adjust(errors, lr)
    return ret

  def init(self):
    self.fc.init()
    self.inited = True

  def getParams(self) -> dict:
    return {
      'inSize': self.fc.shape[0],
      'classCount': self.fc.shape[1]
    }
  def setClassifierPara(self, params: dict):
    self.fc.shape = (params['inSize'], params['classCount'])

class Error(modelObject):
  def __init__(self, x=0, y=0):
    super().__init__(x, y)
    self._init(
      ObjectTypes.Error,
      self.errorProc,
      self.ajErrorProc,
      self.setErrorPara
    )

  def paraString(self):
    return 'Error parameters'
  
  def setDefaultPara(self):
    self.funcPara = 'mse'
    
  def init(self):
    match self.funcPara:
      case 'ce':
        self.func = self.ce
      case 'mae':
        self.func = self.mae
      case 'mse':
        self.func = self.mse
      case _:
        raise Exception('wrong error type')
    self.inited = True
  
  def errorProc(self, data: np.ndarray, label: np.ndarray):
    #label = reshape(label, 2)
    assert data.shape == label.shape, f'data: {data.shape}---label: {label.shape}'
    return self.func(data, label)
  
  def ajErrorProc(self, *args):
    return self.de

  def softmax(self, data):
    data = data - np.max(data, axis=1).reshape(-1, 1)
    data = np.exp(data)
    data = data / np.sum(data, axis=1).reshape(-1, 1)
    return data

  def ce(self, data, label):
    self.de = data - label
    return -np.sum(label * np.log(self.softmax(data) + 1e-8))
  
  def mae(self, data, label):
    self.de = (data > label) * 2 - 1
    return np.sum(np.abs(label - data)) / np.prod(data.shape[:-1])
  
  def mse(self, data, label):
    self.de = data - label
    return np.sum(np.power(label - data, 2)) / np.prod(data.shape[:-1])

  def getParams(self) -> dict:
    return {
      'func': self.funcPara
    }
  
  def setErrorPara(self, params: dict):
    self.funcPara = params['func']

""" class AjConv(modelObject):
  def __init__(self, label, x=0, y=0):
    super().__init__(label, x, y)
    self._init(
      ObjectTypes.AjConv,
      self.AjConvProc,
      self.SetAjConvPara
    )

  def paraString(self):
    return 'AjConv parameters'
  
  def AjConvProc(self):
    pass

  def SetAjConvPara(self):
    pass


class AjFullConn(modelObject):
  def __init__(self, label, x=0, y=0):
    super().__init__(label, x, y)
    self._init(
      ObjectTypes.AjFullConn,
      self.AjFullConnProc,
      self.SetAjFullConnPara
    )

  def paraString(self):
    return 'AjFullConn parameters'
  
  def AjFullConnProc(self):
    pass

  def SetAjFullConnPara(self):
    pass """


class AllModelObj():
  def __init__(self) -> None:
    self.objects: OrderedDict[str, modelObject] = OrderedDict()
    self.connections = AllModelConn()

    self.dataSet: DataSet | None = None
    self.classifier: Classifier | None = None
    self.error: Error | None = None

    self.batchSize = 16
    self.learningRate = 1e-4
    self.iterCount = 10

    self.interruptRequested = False
  
  def __repr__(self) -> str:
    content = ',\n'.join([str(obj) for obj in self.objects.values()])
    return 'AllModelObj{\n' + content + '\n}'
  
  def init(self):
    for obj in self.objects.values():
      obj.init()

  def getParams(self):
    return {
      'batchSize': self.batchSize,
      'learningRate': self.learningRate,
      'iterCount': self.iterCount
    }
  
  def setPara(self, params: dict):
    self.batchSize = params['batchSize']
    self.learningRate = params['learningRate']
    self.iterCount = params['iterCount']
  
  def add(self, obj: modelObject):
    if type(obj) is DataSet:
      if self.dataSet is None:
        self.dataSet = obj
      else:
        raise Exception('can only have one DataSet')
    elif type(obj) is Classifier:
      if self.classifier is None:
        self.classifier = obj
      else:
        raise Exception('can only have one Classifier')
    elif type(obj) is Error:
      if self.error is None:
        self.error = obj
      else:
        raise Exception('can only have one Error')
      
    if obj.id in self.objects.keys():
      raise Exception(f'Object {obj.id} has been added')
    else:
      self.objects[obj.id] = obj

  def procFunc(self):
    assert self.dataSet, 'need DataSet'
    assert self.classifier, 'need Classifier'
    assert self.error, 'need Error'

    total = 0
    correct = 0
    losses = []

    for xb, yb in self.dataSet.loadData(self.batchSize):
      if self.interruptRequested:
        self.interruptRequested = False
        break
      
      obj = self.connections.next(self.dataSet)
      while obj != self.classifier:
        xb = obj.process(xb)
        obj = self.connections.next(obj)
      pred = self.classifier.process(xb)
      loss = self.error.process(pred, yb)
      errs = self.error.adjust(loss, self.learningRate)
      obj = self.classifier
      while obj != self.dataSet:
        errs = obj.adjust(errs, self.learningRate)
        obj = self.connections.prev(obj)

      total += len(xb)
      correct += np.sum(np.argmax(pred, axis=1) == np.argmax(yb, axis=1))
      losses.append(loss)

    testX, testY = self.dataSet.testData()
      
    obj = self.connections.next(self.dataSet)
    while obj != self.classifier:
      testX = obj.process(testX)
      obj = self.connections.next(obj)
    pred = self.classifier.process(testX)
    testAccuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(testY, axis=1)) / len(testY)
    
    return correct / total, testAccuracy, sum(losses) / len(losses)
  
  def predict(self, img: Image.Image):
    assert self.dataSet, 'need DataSet'
    assert self.classifier, 'need Classifier'
    assert self.error, 'need Error'

    x = img.resize(self.dataSet.imgSize)
    x = np.asarray(x)
    x = x.transpose((2,0,1))
    x = (x - x.mean()) / x.std()

    obj = self.connections.next(self.dataSet)
    while obj != self.classifier:
      x = obj.process(x)
      obj = self.connections.next(obj)
    pred = self.classifier.process(x)
    return self.dataSet.labels[np.argmax(pred, axis=1)[0]]
  
  def interrupt(self):
    self.interruptRequested = True



class Connection():
  count = 0

  def __init__(self, s: modelObject, e: modelObject) -> None:
    self.__class__.count += 1
    self.id = f'Line{self.__class__.count}'
    self.NobjS = s
    self.NobjE = e
    self.type = 1 if self.NobjS.type not in (ObjectTypes.AjConv, ObjectTypes.AjFullConn) else 2

  def __repr__(self) -> str:
    return f'<{self.id}, {self.type}, {self.NobjS.id}, {self.NobjE.id}>'


class AllModelConn():
  allowedConns = {
    ObjectTypes.DataSet: (ObjectTypes.Conv,),
    ObjectTypes.Conv: (ObjectTypes.Pool,),
    ObjectTypes.Pool: (ObjectTypes.Conv, ObjectTypes.FullConn),
    ObjectTypes.FullConn: (ObjectTypes.FullConn, ObjectTypes.NonLinear),
    ObjectTypes.NonLinear: (ObjectTypes.Classifier,),
    ObjectTypes.Classifier: (ObjectTypes.Error,),
    ObjectTypes.Error: tuple(),
  }

  def __init__(self) -> None:
    self.connections: typing.Dict[str, Connection] = {}
  
  def __repr__(self) -> str:
    content = ',\n'.join([str(conn) for conn in self.connections.values()])
    return 'AllModelConn{\n' + content + '\n}'

  def add(self, conn: Connection):
    for c in self.connections.values():
      if c.NobjS == conn.NobjS or c.NobjE == conn.NobjE \
        or (c.NobjS == conn.NobjE and c.NobjE == conn.NobjS):
        raise Exception(f'connection conflicted with {c}')

    self.connections[conn.id] = conn

  def __getitem__(self, id):
    return self.connections[id]
  
  def __iter__(self):
    return self.connections.__iter__()

  def remove(self, id):
    self.connections.pop(id)

  def check(self) -> list[Connection]:
    lst = []
    for id, conn in self.connections.items():
      if conn.NobjE.type not in self.allowedConns[conn.NobjS.type]:
        lst.append(conn)
    return lst

 
  def prev(self, obj: modelObject):
    for conn in self.connections.values():
      if conn.NobjE == obj:
        return conn.NobjS
    raise Exception('no prev object')
  
  def next(self, obj: modelObject):
    for conn in self.connections.values():
      if conn.NobjS == obj:
        return conn.NobjE
    raise Exception('no next object')


""" if __name__ == '__main__':

  a = AllModelObj()
  a.add(DataSet())
  a.add(Conv())
  a.add(Pool())
  a.add(FullConn())
  a.add(NonLinear())
  a.add(Classifier())
  a.add(Error())
  print(a)

  b = AllModelConn()
  b.add(Connection(a.objects['DataSet1'], a.objects['Conv1']))
  b.add(Connection(a.objects['Conv1'], a.objects['Pool1']))
  b.add(Connection(a.objects['Pool1'], a.objects['FullConn1']))
  print(b) """