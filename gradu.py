import cv2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from rembg import remove
from torchvision.models.feature_extraction import create_feature_extractor
from numpy import dot
from numpy.linalg import norm
import torch
import os
import requests
import torchvision.transforms as T
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

weights = EfficientNet_B7_Weights.DEFAULT

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 path
protoFile = "D:\\openpose_data\\models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "D:\\openpose_data\\models\\pose\\pose_iter_160000.caffemodel"
 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 이미지 읽어오기
image = cv2.imread("C:\\data\\assemble\\suri.png")
rembg_img = remove(image)
cv2.imwrite("rembg_img.png",rembg_img)
rimg = cv2.imread("rembg_img.png")
# frame.shape = 불러온 이미지에서 height, width, color 받아옴
imageHeight, imageWidth, _ = rimg.shape
 
# network에 넣기위해 전처리
inpBlob = cv2.dnn.blobFromImage(rimg, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=True)
 
# network에 넣어주기
net.setInput(inpBlob)

# 결과 받아오기
output = net.forward()

# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
H = output.shape[2]
W = output.shape[3]
print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

# 키포인트 검출시 이미지에 그려줌
points = []
for i in range(0,15):
    # 해당 신체부위 신뢰도 얻음.
    probMap = output[0, i, :, :]
    
    # global 최대값 찾기
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 원래 이미지에 맞게 점 위치 변경
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
    if prob > 0.1 :    
        cv2.circle(rimg, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
        cv2.putText(rimg, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)
print(points)
topY = points[1][1]
topX = points[5][0]
bottomY = points[11][1]
bottomX = points[3][0]
topImage = rimg[topY:bottomY,bottomX-20:topX+20]
bottomImage = rimg[bottomY:imageHeight,bottomX-20:topX+20]

cv2.imwrite("D:\\data\\topImage.png",topImage)
cv2.imwrite("D:\\data\\bottomImage.png",bottomImage)

cv2.imwrite("D:\\data\\posePointImage.png",rimg)



imageCopy = rimg

for pair in POSE_PAIRS:
    partA = pair[0]             # Head
    partA = BODY_PARTS[partA]   # 0
    partB = pair[1]             # Neck
    partB = BODY_PARTS[partB]   # 1
    
    #print(partA," 와 ", partB, " 연결\n")
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)

cv2.imwrite("D:\\data\\poseLineImage.png",imageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

model = efficientnet_b7(weights=weights)
model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
model.eval()

path = "C:\\data\\assemble"
source_top = "D:\\data\\topImage.png"
source_bottom = "D:\\data\\bottomImage.png"
image_files1 = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]

similarities = []
similarities2 = []

path_list = []
histo = []
combine_similar = []

def predict(img):
    resized_image = image_resize(img)
    predicted_result = model(resized_image)
    image_feature = torch.flatten(predicted_result['avgpool'])
    return image_feature.detach().numpy()

def image_resize(image_url):
    image = Image.open(image_url)
    rgb_image = image.convert('RGB')
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()]
    )
    return preprocess(rgb_image).unsqueeze(0)

def extract_color_histogram(image_path, clusters=4):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=clusters, n_init=10)
    clt.fit(image)
    hist = centroid_histogram(clt)
    return clt.cluster_centers_, hist

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist
def compare_images(image1_path, image2_path):
    # 이미지 불러오기
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 이미지 크기 조정
    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))

    # 이미지를 그레이스케일로 변환
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 히스토그램 계산
    hist1 = cv2.calcHist([image1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2_gray], [0], None, [256], [0, 256])

    # 히스토그램 비교
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity

def find_secondary_color(clusters, hist):
    sorted_clusters = sorted(zip(hist, clusters), reverse=True)
    return sorted_clusters[1][1] if len(sorted_clusters) > 1 else sorted_clusters[0][1]

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

sourceTop_embedding = predict(source_top)
sourceBottom_embedding = predict(source_bottom)
for img_file1 in image_files1:
    img = predict(img_file1)
    cs = cos_sim(img,sourceTop_embedding)
    cs2 = cos_sim(img,sourceBottom_embedding)

    path_list.append(img_file1)
    # histo = compare_images(img_file1,source_top)
    # histo2 = compare_images(img_file1,source_bottom)

    combine_similar = (cs)
    combine_similar2 = (cs2)
    print(cs,cs2,img_file1)
    similarities.append(combine_similar)
    similarities2.append(combine_similar2)


pairs = list(zip(similarities,path_list))
pairs.sort(reverse=True)

pairs2 = list(zip(similarities2,path_list))
pairs2.sort(reverse=True)

fig = plt.figure(figsize=(8,6))


for i in range(3):
    plt.subplot(2,4,i+1)
    
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
    
    plt.title(f'Bottom {i}')
    plt.xticks([])
    plt.yticks([])
    
plt.subplot(2,4,8)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('source')
plt.xticks([])
plt.yticks([])
plt.show()