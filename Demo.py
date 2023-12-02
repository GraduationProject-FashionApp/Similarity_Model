import cv2
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

from rembg import remove
from torchvision.models.feature_extraction import create_feature_extractor
from numpy import dot
from numpy.linalg import norm
import torch
import os
import torchvision.transforms as T
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

weights = EfficientNet_B3_Weights.DEFAULT

similarities = []
similarities2 = []
path_list = []
path_list2 = []
histo = []
combine_similar = []
searchSimilar = []
searchPath = []
searchSimilar2 = []
searchPath2 = []

wholePath = './.data/assemble2'

bottomPath = './.data/assemble4'
topPath = './.data/assemble5'
# 이미지 읽어오기
image = cv2.imread("suri.png")

# -- remBg 
rembg_img = remove(image)
cv2.imwrite("rembg_img.png",rembg_img)

rimg = cv2.imread("rembg_img.png")
# frame.shape = 불러온 이미지에서 height, width, color 받아옴
 
# -- pose estimate
# 각 파일 path
protoFile = "openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "openpose-master/models/pose/pose_iter_160000.caffemodel"
 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
imageHeight, imageWidth, _ = rimg.shape

# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(rimg, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 
# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

H = output.shape[2]
W = output.shape[3]

points = []
for i in range(0,15):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H
    if prob > 0.05 :    
        points.append((int(x), int(y)))
    else :
        points.append(None)
# Pose Estimate로 받아온 값으로 좌표 설정
topY = points[1][1]
topX = points[5][0]
bottomY = points[11][1]
bottomX = points[3][0]
topImage = rimg[topY:bottomY,bottomX-20:topX+20]
bottomImage = rimg[bottomY+10:imageHeight,bottomX-20:topX+20]

cv2.imwrite('posePointImage.png',rimg)

cv2.imwrite('topImage.png',topImage)
cv2.imwrite('bottomImage.png',bottomImage)

imageCopy = rimg

# -- efficientnet
model = efficientnet_b3(weights=weights)
model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
model.eval()

bottomFile = [os.path.join(bottomPath, f) for f in os.listdir(bottomPath) if f.endswith(('.png', '.jpg', '.jpeg'))]
topFile = [os.path.join(topPath, f) for f in os.listdir(topPath) if f.endswith(('.png', '.jpg', '.jpeg'))]

def predict(img):
    resized_image = image_resize(img)
    predicted_result = model(resized_image)
    image_feature = torch.flatten(predicted_result['avgpool'])
    return image_feature.detach().numpy()

def image_resize(image_url):
    image = Image.open(image_url)
    rgb_image = image.convert('RGB')
    preprocess = T.Compose([
        T.Resize(300, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()]
    )
    return preprocess(rgb_image).unsqueeze(0)

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))
source_top = "topImage.png"
source_bottom = "bottomImage.png"
sourceTop_embedding = predict(source_top)
sourceBottom_embedding = predict(source_bottom)
for file in bottomFile:
    img = predict(file)
    cs = cos_sim(img,sourceBottom_embedding)
    searchPath.append(file)

    combine_similar = (cs)
    searchSimilar.append(combine_similar)

bottomPairs = list(zip(searchSimilar,searchPath))
bottomPairs.sort(reverse=True)
target = bottomPairs[0][1]
bottomFolder = ''
if("Denim" in target):
    bottomFolder = "Denim"
elif("Slacks" in target):
    bottomFolder = "Slacks"
elif("Short" in target):
    bottomFolder = "Short"
elif("Training" in target):
    bottomFolder = "Training"
elif("Cotton" in target):
    bottomFolder = "Cotton"
print(bottomFolder)
path = './.data/each/'+bottomFolder
image_files1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_file1 in image_files1:
    img = predict(img_file1)
    cs2 = cos_sim(img,sourceBottom_embedding)

    path_list2.append(img_file1)
    combine_similar2 = (cs2)
    similarities2.append(combine_similar2)

for file in topFile:
    img = predict(file)
    cs = cos_sim(img,sourceTop_embedding)

    searchPath2.append(file)
    combine_similar = (cs)
    searchSimilar2.append(combine_similar)

topPairs = list(zip(searchSimilar2,searchPath2))
topPairs.sort(reverse=True)
target = topPairs[0][1]

topFolder = ''
if("Hood" in target):
    topFolder = "Hood"
elif("TShirt" in target):
    topFolder = "TShirt"
elif("Shirts" in target):
    topFolder = "Shirts"
elif("Sweatshirt" in target):
    topFolder = "Sweatshirt"
elif("Sleeveless" in target):
    topFolder = "Sleeveless"
print(topFolder)
path = './.data/each/'+topFolder

image_files2 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_file2 in image_files2:
    img = predict(img_file2)
    cs = cos_sim(img,sourceTop_embedding)

    path_list.append(img_file2)
    combine_similar = (cs)
    similarities.append(combine_similar)

pairs = list(zip(similarities,path_list))
pairs.sort(reverse=True)

pairs2 = list(zip(similarities2,path_list2))
pairs2.sort(reverse=True)

# -- 결과값 출력
topTen = []
bottomTen = []

fig = plt.figure(figsize=(8,6))
for i in range(10):
    print(pairs[i],pairs2[i])
    topTen.append(pairs[i])
    bottomTen.append(pairs2[i])

print(topTen)
print(bottomTen)

for i in range(3):
    plt.subplot(2,4,i+1)
    plt.imshow(cv2.cvtColor(cv2.imread(pairs[i][1]),cv2.COLOR_BGR2RGB))
    plt.title(f'Top {i}')
    plt.xticks([])
    plt.yticks([])
    
plt.subplot(2,4,4)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('source')
plt.xticks([])
plt.yticks([])
for i in range(3):
    plt.subplot(2,4,i+5)
    plt.imshow(cv2.cvtColor(cv2.imread(pairs2[i][1]),cv2.COLOR_BGR2RGB))
    plt.title(f'Bottom {i}')
    plt.xticks([])
    plt.yticks([])
    
plt.subplot(2,4,8)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('source')
plt.xticks([])
plt.yticks([])
plt.show()
