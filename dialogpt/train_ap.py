import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from dataloader import TextDataset
from model_rg import DialoGPT
from utils.model_utils import loop, get_device_ids, CustomPaddingTensorCollator, set_max_len
import sys
from tqdm import tqdm

transformer_model = 'microsoft/DialoGPT-small'
input_format = 'action_prediction'
# input_format = 'action_prediction_single'
model_name = f'{input_format}_{transformer_model.replace("/", "#")}'
#cuda = True
cuda=False
mixed_precision = False
clear_cache_every = 200

device_ids = get_device_ids(cuda=cuda)
print('Using devices: {}'.format(device_ids))

#BATCH_SIZE = int(sys.argv[1])
BATCH_SIZE = 64
print('Batch size: {}'.format(BATCH_SIZE))
MINI_BATCH_SIZE = min((len(device_ids) << 4) if cuda else 16, BATCH_SIZE)
print('Mini batch size for training: {}'.format(MINI_BATCH_SIZE))
BATCH_SIZE_EVAL = min((len(device_ids) << 5) if cuda else 32, BATCH_SIZE << 1)
print('Batch size for evaluation: {}'.format(BATCH_SIZE_EVAL))

assert BATCH_SIZE % MINI_BATCH_SIZE == 0
batch_iters = BATCH_SIZE // MINI_BATCH_SIZE

workers = max(min(16, MINI_BATCH_SIZE >> 3), 4)
workers_eval = max(min(8, BATCH_SIZE_EVAL >> 3), 4)


train = TextDataset.create_data(f'resources/train.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), shuffle=False)
val = TextDataset.create_data(f'resources/val.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), training=False, shuffle=False)
test = TextDataset.create_data(f'resources/test.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), training=False, shuffle=False)

set_max_len(train, val, test)

pad_value = train.tokenizer.pad_token_id
if pad_value is None:
    pad_value = train.tokenizer.eos_token_id

key2pad_id = {
    'input_ids': pad_value,
    'labels': pad_value
}

first_eval = {
    'input_ids': True,
    'attention_mask': True
}

collator_train = CustomPaddingTensorCollator(key2pad_id=key2pad_id)
collator_eval = CustomPaddingTensorCollator(key2pad_id=key2pad_id, first=first_eval)

dataloader_train = DataLoader(train, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True, collate_fn=collator_train)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collator_eval)
dataloader_test = DataLoader(test, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collator_eval)

epochs = 25
epochs_per_val = 1
epochs_per_test = 1
min_epoch_for_val = epochs * 0.2
min_epoch_for_test = epochs * 0.2
steps = epochs * len(dataloader_train)

#model = DialoGPT(transformer_model, num_training_steps=steps, lr=5e-5, device_idxs=device_ids, mixed_precision=mixed_precision, cuda=cuda, pad_value=pad_value)
model = DialoGPT(transformer_model, num_training_steps=steps, lr=5e-5, device_idxs=device_ids, cuda=cuda, pad_value=pad_value)
# model.load_model('checkpoint/action_prediction_microsoft#DialoGPT-small_13-01-2021 15:43:55.1610523835_val_acc-AR:0.371_val_loss-L:5.33 LR:5.325.th')

if cuda:
    # if len(device_ids) > 1:
    #     model = DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
    model.to('cuda:' + str(device_ids[0]))

pbar = tqdm(range(epochs))
for i in pbar:
    pbar.set_description(f'Epoch {i + 1}/{epochs}')
    loop(model, dataloader_train, batch_iters=batch_iters, clear_cache_every=clear_cache_every, train=True, cuda=cuda, model_name=model_name)
    if (i + 1) >= min_epoch_for_val and (i + 1 - min_epoch_for_val) % epochs_per_val == 0:
        loop(model, dataloader_val, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name)
    if (i + 1) >= min_epoch_for_test and (i + 1 - min_epoch_for_val) % epochs_per_test == 0:
        loop(model, dataloader_test, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, save_best=False, save_results=True)
        # loop(model, dataloader_test, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, save_best=True, save_results=True)
