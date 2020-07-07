from PIL import Image
import numpy as np
img = Image.open('F:\Python_Workspace\PyTorch\cat.jpg')
print(img.size)
x = np.array(img)
print(x)