from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
sys.path.append('./utils')

from generic_utils import read
import os
import random
import re
from torchvision.transforms import transforms
from PIL import Image
from copy import deepcopy

all_imgs = {file: os.path.join(root, file) for root, _, files in os.walk('/storage/longhl/mmdial_models/data/images') for file in files if file.lower().endswith(('.jpg', '.png', '.jpeg'))}

token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')


def url2img(url_or_img):
    begin_idx = url_or_img.rfind('/')
    if begin_idx != -1:
        return url_or_img[begin_idx + 1:]
    return url_or_img


def has_token(text):
    return next_token(text) is not None


def next_token(text):
    result = token_matcher.search(text)
    return result if result is None else result[0]


def get_token_text(token):
    return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')


def extract(text, begin_token, end_token=None, no_token_in_between=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return '', None
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return '', None
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        return text[begin_with_len: end_idx].strip(), begin_idx
    recurse_result = extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
    return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None


def remove(text, begin_token, end_token=None, no_token_in_between=True, remove_begin_token=True, remove_end_token=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return text
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return text
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        end_with_len = end_idx + len(end_token)
        return text[:(begin_idx if remove_begin_token else begin_with_len)].strip() + ' ' + text[(end_with_len if remove_end_token else end_idx):].strip()
    return text[:begin_with_len] + remove(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between, remove_begin_token=remove_begin_token, remove_end_token=remove_end_token)


class BaseDataset(Dataset):
    def __init__(self, data, training, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.training = training

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        raise NotImplementedError()

    @classmethod
    def read_data(cls, paths):
        raise NotImplementedError()

    @classmethod
    def create_data(cls, paths, split=(1,), training=(True,), shuffle=True, **kwargs):
        assert sum(split) == 1
        if isinstance(training, bool):
            training = [training for _ in range(len(split))]
        elif len(training) == 1 and len(split) > 1:
            training = [training[0] for _ in range(len(split))]
        assert len(split) == len(training)
        if isinstance(paths, str):
            paths = [paths]
        print('Preparing data...')
        data = cls.read_data(paths)
        if shuffle:
            random.shuffle(data)
        splits = []
        begin_idx = 0
        for i, s in enumerate(split):
            if i == len(split) - 1:
                end_idx = len(data)
            else:
                end_idx = int(begin_idx + len(data) * s)
            splits.append(cls(data[begin_idx: end_idx], training=training[i], **kwargs))
            begin_idx = end_idx
        return splits[0] if len(split) == 1 else splits


class TextDataset(BaseDataset):
    def __init__(
        self,
        data,
        training,
        tokenizer_or_transformer_model,
        max_len=1024,
        end_token=' <|endofresponse|>',
        remove_tokens={
            '<|imagesource|>': {'<|system|>', '<|user|>', '<|endofcontext|>', '<|endofresponse|>'}
        },
        split_token='<|endofcontext|>',
        **kwargs
    ):
        super().__init__(data, training, **kwargs)
        if isinstance(tokenizer_or_transformer_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_transformer_model)
        else:
            self.tokenizer = tokenizer_or_transformer_model
        self.end_tokens = self.tokenizer(end_token, truncation=True, max_length=1024, return_tensors='pt')['input_ids']
        self.remove_tokens = remove_tokens
        self.split_token = split_token
        self.max_len = max_len

    @classmethod
    def read_data(cls, paths):
        data = []
        for path in tqdm(paths):
            data.extend(read(path))
        return data

    def __getitem__(self, i):
        raw_sample = self.data[i]
        for remove_token, end_tokens in self.remove_tokens.items():
            end_tokens = deepcopy(end_tokens)
            while end_tokens:
                for end_token in list(end_tokens):
                    img_src, _ = extract(raw_sample, remove_token, end_token=end_token)
                    if not img_src:
                        end_tokens.discard(end_token)
                    else:
                        raw_sample = remove(raw_sample, remove_token, end_token=end_token, remove_end_token=False)
        split_idx = raw_sample.rindex(self.split_token) + len(self.split_token)
        if self.training:
            sample = self.tokenizer(raw_sample, truncation=True, max_length=1024, return_tensors='pt')
        else:
            sample = self.tokenizer(raw_sample[:split_idx], truncation=True, max_length=1024, return_tensors='pt')
        sample['labels'] = self.tokenizer(raw_sample[split_idx:], truncation=True, max_length=1024, return_tensors='pt')['input_ids']
        sample['labels_begin'] = sample['input_ids'].shape[-1] - sample['labels'].shape[-1]
        sample['labels_end'] = sample['input_ids'].shape[-1]
        sample['id'] = i
        sample['end_tokens'] = self.end_tokens
        sample['max_len'] = self.max_len
        return sample


# The part below is WIP
class ImageDataset(BaseDataset):
    def __init__(self, data, training, image_size,
                 mean = [0.4920555, 0.4224293, 0.34641075],
                 std = [0.2283078, 0.22627847, 0.22060438],
                 **kwargs):
        super().__init__(data, training, **kwargs)
        transforms.Compose([
            transforms.Resize(round(image_size / 0.875), interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # self.augmentor = augmentor

    @classmethod
    def read_data(cls, paths):
        ...


class ImageClassificationDataset(ImageDataset):
    def __getitem__(self, i):
        sample = self.data[i]
        img = Image.open(sample[0]).convert('RGB')
        label = sample[1]
        if self.return_file:
            return img, label, sample[2]
        return img, label


if __name__ == '__main__':
    test = ResponseGenerationDataset.create_data('resources/train.dialogpt', 'microsoft/DialoGPT-small', split=(1,), training=False, shuffle=False)
    print(test[15359])
