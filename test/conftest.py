import pytest
from PIL import Image
import os
import torchvision
from torchvision import transforms
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from noisegrad.explainers import saliency_explainer, explain_gradient_x_input
import torch


@pytest.fixture(scope="session")
def label():
    return torch.tensor([485])


@pytest.fixture(scope="session")
def text_labels():
    return torch.zeros(size=(8,))


@pytest.fixture(scope="session")
def normalised_image():
    image = Image.open("samples/llama.jpg")

    # Transform image to a 224x224 image and normalise pixel values
    nr_channels = 3
    img_size = 224
    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    normalised_image = transformations(image).reshape(
        1, nr_channels, img_size, img_size
    )
    return normalised_image


@pytest.fixture(scope="session")
def resnet18_model():
    return torchvision.models.resnet18(pretrained=True)


@pytest.fixture(scope="session")
def baseline_explanation(resnet18_model, normalised_image, label):
    return saliency_explainer(resnet18_model, normalised_image, label)


@pytest.fixture(scope="session")
def distilbert_model():
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


@pytest.fixture(scope="session")
def distilbert_tokenizer():
    return AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


@pytest.fixture(scope="session")
def text_batch():
    return load_dataset("sst2")["test"]["sentence"][:8]


@pytest.fixture(scope="session")
def input_ids(distilbert_tokenizer, text_batch):
    return distilbert_tokenizer(text_batch, padding="longest", return_tensors="pt")[
        "input_ids"
    ]


@pytest.fixture(scope="session")
def attention_mask(distilbert_tokenizer, text_batch):
    return distilbert_tokenizer(text_batch, padding="longest", return_tensors="pt")[
        "attention_mask"
    ]


@pytest.fixture(scope="session")
def input_embeddings(distilbert_model, input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.int64, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    word_embeddings = distilbert_model.get_input_embeddings()(input_ids)
    position_embeddings = distilbert_model.get_position_embeddings()(position_ids)
    return word_embeddings + position_embeddings


@pytest.fixture(scope="session")
def baseline_explanation_text(
    distilbert_model, input_embeddings, attention_mask, text_labels
):
    return explain_gradient_x_input(
        distilbert_model, input_embeddings, text_labels, attention_mask
    )
