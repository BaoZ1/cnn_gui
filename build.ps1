python -m nuitka `
  --include-data-files=D:/miniconda3/envs/labCNN/Lib/site-packages/PyQt6\Qt6/plugins/platforms/*=platforms/ `
  --include-data-files=D:/miniconda3/envs/labCNN/Lib/site-packages/PyQt6\Qt6/plugins/styles/*=styles/ `
  --include-data-files=./icons/*=icons/ `
  --standalone --disable-console main.py