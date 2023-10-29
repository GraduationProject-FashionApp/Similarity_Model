from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)
model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
model.eval()

import requests
import torchvision.transforms as T
from PIL import Image

source_url = "test1.png"
target_url = "test2.png"
not_same_target_url = "test3.jpg"


def image_resize(image_url):
    image = Image.open(image_url)
    rgb_image = image.convert('RGB')
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()]
    )
    print(preprocess(rgb_image).size())
    return preprocess(rgb_image).unsqueeze(0)


from numpy import dot
from numpy.linalg import norm
import torch


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def predict(image_url):
    resized_image = image_resize(image_url)
    predicted_result = model(resized_image)
    image_feature = torch.flatten(predicted_result['avgpool'])
    return image_feature.detach().numpy()


source_embedding = predict(source_url)
target_embedding = predict(target_url)

print(cos_sim(source_embedding, target_embedding))
not_same_target_embedding = predict(not_same_target_url)
print(cos_sim(source_embedding, not_same_target_embedding))