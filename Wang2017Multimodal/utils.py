import torchvision.transforms as transforms
import os


""" Saves image in the /output folder with a specified name as .jpg """


def save_image(tensor, title="output"):
    image = tensor.cpu().clone()  # clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    denormalizer = tensor_denormalizer()
    image = denormalizer(image)
    image.data.clamp_(0, 1)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    scriptDir = os.path.dirname(__file__)
    image.save(title)


""" Transforms to normalize the image while transforming it to a torch tensor """


def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


""" Denormalizes image to save or display it """


def tensor_denormalizer():
    return transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
