import torch
import torchvision
from tqdm.auto import tqdm
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def evaluate(model, dataloader, device):
    model.eval()
    total_acc = []
    total_loss = []
    running_acc = 0.0
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        probabilities = F.sigmoid(outputs)
        predicted_labels = (probabilities >= 0.5).float()
        running_acc += (predicted_labels == labels).sum().item() / labels.numel()

    return running_acc / len(dataloader) * 100



def predict(model, image_path, threshold, transforms, labels, device):

    dict_labels = {0: '5_o_Clock_Shadow',
                    1: 'Arched_Eyebrows',
                    2: 'Attractive',
                    3: 'Bags_Under_Eyes',
                    4: 'Bald',
                    5: 'Bangs',
                    6: 'Big_Lips',
                    7: 'Big_Nose',
                    8: 'Black_Hair',
                    9: 'Blond_Hair',
                    10: 'Blurry',
                    11: 'Brown_Hair',
                    12: 'Bushy_Eyebrows',
                    13: 'Chubby',
                    14: 'Double_Chin',
                    15: 'Eyeglasses',
                    16: 'Goatee',
                    17: 'Gray_Hair',
                    18: 'Heavy_Makeup',
                    19: 'High_Cheekbones',
                    20: 'Male',
                    21: 'Mouth_Slightly_Open',
                    22: 'Mustache',
                    23: 'Narrow_Eyes',
                    24: 'No_Beard',
                    25: 'Oval_Face',
                    26: 'Pale_Skin',
                    27: 'Pointy_Nose',
                    28: 'Receding_Hairline',
                    29: 'Rosy_Cheeks',
                    30: 'Sideburns',
                    31: 'Smiling',
                    32: 'Straight_Hair',
                    33: 'Wavy_Hair',
                    34: 'Wearing_Earrings',
                    35: 'Wearing_Hat',
                    36: 'Wearing_Lipstick',
                    37: 'Wearing_Necklace',
                    38: 'Wearing_Necktie',
                    39: 'Young'}

    train_mean =  torch.tensor([0.5065, 0.4260, 0.3833])
    train_std = torch.tensor([0.3078, 0.2876, 0.2871])
    
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    image = transforms(image)
    image = image.to(device).unsqueeze_(dim=0)
    
    output = model(image).squeeze().detach().cpu()# .tolist()
    probabilities = F.sigmoid(output)
    predicted_labels = (probabilities >= threshold).squeeze().int().tolist()
    print('Predicted Labels:')
    print('-------------\n')
    for idx, label in enumerate(predicted_labels):
        if label:
            print(dict_labels[idx])

    image_name = image_path.split('/')[3]

    print('\n\nTrue Labels:')
    print('-------------\n')
    for idx, label in enumerate(labels.loc[image_name].tolist()):
        if label:
            print(dict_labels[idx])
