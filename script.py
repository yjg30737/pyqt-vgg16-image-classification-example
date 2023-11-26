import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG11_Weights

classes = VGG11_Weights.IMAGENET1K_V1.meta['categories']

class ImagePredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = models.vgg16(weights='IMAGENET1K_V1')
        torch.save(self.model.state_dict(), model_path)
        self.transform = self.load_transform()

    def load_model(self, model_path):
        self.model = models.vgg16()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def load_transform(self):
        img_height = 224
        img_width = 224

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])
        return self.transform

    def get_image_from_url(self, image_url):
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image

    def predict_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)  # 배치 차원 추가
        image = image.to(self.device)
        #
        output = self.model(image)
        _, predicted = torch.max(output, 1)

        prob = F.softmax(output, dim=1)[0] * 100

        # get top 5 highest prediction entries
        probs = [round(prob.item(), 2) for prob in prob]
        predictions = dict(zip(classes, probs))

        top_5_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]

        result_label = ''
        for i, prediction in enumerate(top_5_predictions):
            lbl = prediction[0]
            pred = prediction[1]
            result_label += f'{i+1}: <span style="color: blue">Predicted: {lbl}</span><br/>' \
               f'Percent: {pred}<br/><br/>'
        return result_label