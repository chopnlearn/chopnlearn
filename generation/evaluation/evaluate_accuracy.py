import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import (BaseCLIPObjectStateClassifierDataset,
                     CLIPObjectStateImageFolderDataset, collate_fn_clip)
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel


class CLIPObjectStateClassifier(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.model = CLIPModel.from_pretrained(args.pretrained_model)

        self.train_text_model = args.train_text_model

        self.model.requires_grad_(True)
        self.model.logit_scale.requires_grad_(False)
        if self.train_text_model:
            self.trainable_params = list(self.model.text_model.parameters()) + list(self.model.vision_model.parameters()) + list(self.model.text_projection.parameters()) + list(self.model.visual_projection.parameters())
        else:
            self.trainable_params = list(self.model.vision_model.parameters()) + list(self.model.visual_projection.parameters())
            self.model.text_model.requires_grad_(False)
            self.model.text_projection.requires_grad_(False)
            

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.accuracy = lambda logits, targets: (logits.argmax(-1) == targets).float().mean()

        self.best_val_accu = 0.0
        self.best_test_accu = 0.0


    def build_labels(self, dataset: BaseCLIPObjectStateClassifierDataset):
        self.object_id2label = dataset.object_id2label
        self.state_id2label = dataset.state_id2label
        self.object_label2id = dataset.object_label2id
        self.state_label2id = dataset.state_label2id

        if self.train_text_model:
            self.object_text_ids = [dataset.text_ids_dict[self.object_id2label[i]] for i in range(len(self.object_id2label))]
            self.state_text_ids = [dataset.text_ids_dict[self.state_id2label[i]] for i in range(len(self.state_id2label))]

            self.object_text_ids = torch.stack(self.object_text_ids, dim=0).to(self.device)
            self.state_text_ids = torch.stack(self.state_text_ids, dim=0).to(self.device)

        else:
            self.object_text_features = [dataset.text_feature_dict[self.object_id2label[i]] for i in range(len(self.object_id2label))]
            self.state_text_features = [dataset.text_feature_dict[self.state_id2label[i]] for i in range(len(self.state_id2label))]
            self.object_text_features = torch.stack(self.object_text_features, dim=0).to(self.device)
            self.state_text_features = torch.stack(self.state_text_features, dim=0).to(self.device)


    def forward(self, pixel_values):
        if self.train_text_model:
            label_object_ids = self.object_text_ids
            label_state_ids = self.state_text_ids
            label_object_features = self.model.get_text_features(label_object_ids.to(self.device))
            label_object_features = label_object_features / label_object_features.norm(p=2, dim=-1, keepdim=True)
            label_state_features = self.model.get_text_features(label_state_ids.to(self.device))
            label_state_features = label_state_features / label_state_features.norm(p=2, dim=-1, keepdim=True)
        else:
            label_object_features = self.object_text_features.to(self.device)
            label_state_features = self.state_text_features.to(self.device)

        image_features = self.model.get_image_features(pixel_values)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        object_logits = torch.matmul(image_features, label_object_features.t())
        state_logits = torch.matmul(image_features, label_state_features.t())

        return {
            'object_logits': object_logits,
            'state_logits': state_logits,
            'image_features': image_features, 
            'label_object_features': label_object_features,
            'label_state_features': label_state_features,
        }


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        return [optimizer], [lr_scheduler]


    def training_step(self, batch, batch_idx):

        pixel_values = batch["pixel_values"]
        object_ids = batch["object_ids"]
        state_ids = batch["state_ids"]

        outputs = self(pixel_values)
        
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        object_logits = outputs['object_logits'] * logit_scale
        states_logits = outputs['state_logits'] * logit_scale

        object_loss = self.loss_func(object_logits, object_ids)
        state_loss = self.loss_func(states_logits, state_ids)

        object_accu = self.accuracy(object_logits, object_ids)
        state_accu = self.accuracy(states_logits, state_ids)

        loss = object_loss + state_loss
        avg_accu = (object_accu + state_accu) / 2

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_object_loss", object_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_state_loss", state_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_object_accu", object_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_state_accu", state_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_avg_accu", avg_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        
        pixel_values = batch["pixel_values"]   
        object_ids = batch["object_ids"]
        state_ids = batch["state_ids"]
        outputs = self(pixel_values)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        object_logits = outputs['object_logits'] * logit_scale
        state_logits = outputs['state_logits'] * logit_scale

        object_loss = self.loss_func(object_logits, object_ids)
        state_loss = self.loss_func(state_logits, state_ids)

        object_accu = self.accuracy(object_logits, object_ids)
        state_accu = self.accuracy(state_logits, state_ids)

        loss = object_loss + state_loss
        avg_accu = (object_accu + state_accu) / 2

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_object_loss", object_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_state_loss", state_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_object_accu", object_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_state_accu", state_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_avg_accu", avg_accu, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return {'val_loss': loss}
    

    @torch.no_grad()
    def evaluate_on(self, dataset: BaseCLIPObjectStateClassifierDataset, batch_size=16, num_workers=4):
        self.eval()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda items: collate_fn_clip(items), num_workers=num_workers)
        accuracy_func = self.accuracy
        self.cuda()

        details = []

        object_acc_list = []
        state_acc_list = []
        for batch in tqdm(dataloader):
            outputs = self(pixel_values=batch['pixel_values'].cuda())

            object_logits = outputs['object_logits']
            state_logits = outputs['state_logits']


            object_acc_list.append(accuracy_func(object_logits, batch['object_ids'].cuda()))
            state_acc_list.append(accuracy_func(state_logits, batch['state_ids'].cuda()))

            for objext_logit, state_logit, object_label, state_label in zip(outputs['object_logits'], outputs['state_logits'], batch['object_ids'], batch['state_ids']):
                details.append(
                    {
                        'object_label': dataset.object_id2label[object_label.item()],
                        'object_pred': dataset.object_id2label[objext_logit.argmax(-1).item()],
                        'state_label': dataset.state_id2label[state_label.item()],
                        'state_pred': dataset.state_id2label[state_logit.argmax(-1).item()],
                    }
                )
            
        object_acc = torch.stack(object_acc_list).mean().item()
        state_acc = torch.stack(state_acc_list).mean().item()

        return object_acc, state_acc, details



def get_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_ckpt_path', type=str, required=True)
    parser.add_argument('--path_gen', type=str, required=True)
    parser.add_argument('--data', type=str, default='../image_dataset/images')
    parser.add_argument('--split', type=str, default='../image_dataset/test_split.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='evaluation/accuracy_results')
    parser.add_argument('--exclude_states', nargs='+', type=str, default=['p'])
    args = parser.parse_args(input_args) if input_args else parser.parse_args()

    args.data_dir = os.path.dirname(args.data)
    args.data_folder = os.path.basename(args.data)

    return args


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available'

    classifier = CLIPObjectStateClassifier.load_from_checkpoint(args.classifier_ckpt_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(classifier.args.pretrained_model)

    normalize_mean = [0.48145466, 0.4578275, 0.40821073]
    normalize_std = [0.26862954, 0.26130258, 0.27577711]

    dataset = CLIPObjectStateImageFolderDataset(
        image_folder=args.path_gen,
        data_root=args.data_dir,
        data_folder=args.data_folder,
        tokenizer=tokenizer,
        split=args.split,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]),
        clip_model=classifier.model,
        exclude_states = args.exclude_states,
        return_text_features=not classifier.train_text_model,
    )

    classifier.build_labels(dataset)

    object_acc, state_acc, details = classifier.evaluate_on(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f'Object acc: {object_acc}, state acc: {state_acc}')

    os.makedirs(args.output_dir, exist_ok=True)

    to_be_saved = {
        'object_acc': object_acc,
        'state_acc': state_acc,
        # 'details': details,
    }

 
    json_path = os.path.join(args.output_dir, f'{os.path.basename(args.path_gen)}_acc.json')
    with open(json_path, 'w') as f:
        json.dump(to_be_saved, f, indent=4)

    # save details to a csv file
    csv_path = os.path.join(args.output_dir, f'{os.path.basename(args.path_gen)}_details.csv')
    df_details = pd.DataFrame(details)
    df_details.to_csv(csv_path, index=True)



if __name__ == '__main__':
    args = get_args()
    main(args)


