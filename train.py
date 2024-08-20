import os, json, yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.files import load_yaml_data, load_json_data
from models.vqa_stack_model import VQAStackModel
from dataloader.vqa_dataset import VQADataset

class Trainer:
    def __init__(self, transform=None, config_path="config.yml"):
        self.config = load_yaml_data(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transform

        self.labels = load_json_data(self.config.get("most_common_answers_file_path"))
        self.model = VQAStackModel(num_classes=self.config.get("num_classes")).to(self.device)
        # self.load_checkpoint(self.config.get("checkpoint_path"))

        self.train_dataset = VQADataset(folder=self.config.get("train_folder_path"), transform=self.transform)
        self.val_dataset = VQADataset(folder=self.config.get("val_folder_path"), transform=self.transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.get("batch_size", 2048), shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.get("batch_size", 2048), shuffle=False)

        self.configure_optimizer()
        self.max_epochs = self.config.get('max_epochs', 1000)
        self.best_test_loss = 1e6

        self.writer = SummaryWriter(log_dir=self.config.get('log_dir', 'runs'))

    def label_encode(self, answer):
      if answer in self.labels.keys():
        return self.labels[answer]
      return self.config.get("num_classes")-1

    def fit(self):
        for epoch in range(0, self.max_epochs + 1):
            print("TRAIN EPOCH")
            self.train_epoch(epoch)
            print("VALID EPOCH")
            self.validate_epoch(epoch)
        self.close_writer()

    def train_epoch(self, epoch):
      self.model.train()
      losses = []
      correct_1, total_1 = 0, 0
      correct_5, total_5 = 0, 0

      for image_ebd, question_ebd, answer in tqdm(self.train_dataloader):
          image_ebd = image_ebd.to(self.device)
          question_ebd = question_ebd.to(self.device)
          answer_labels = torch.tensor([self.label_encode(ans) for ans in answer], dtype=torch.long).to(self.device)

          self.optimizer.zero_grad()

          preds = self.model(question_ebd, image_ebd).squeeze(1)
          loss = F.cross_entropy(preds, answer_labels)
          loss.backward()
          self.optimizer.step()
          losses.append(loss.item())
          self.writer.add_scalar("Loss/train", loss.item(), epoch)

          preds = F.softmax(preds, dim=1)
          correct_1_inc, total_1_inc = self.evaluate_accuracy(preds, 1, answer_labels)
          correct_1 += correct_1_inc
          total_1 += total_1_inc

          correct_5_inc, total_5_inc = self.evaluate_accuracy(pred=preds, k=5, answer=answer_labels)
          correct_5 += correct_5_inc
          total_5 += total_5_inc

      mean_train_loss = sum(losses) / len(losses)
      top1_acc = correct_1 / total_1 * 100
      top5_acc = correct_5 / total_5 * 100

      if epoch % 5 == 0:
          print(f"TRAIN {epoch}, {mean_train_loss}")
          print(f"Top-{1} Accuracy: {top1_acc:.2f}%")
          print(f"Top-{5} Accuracy: {top5_acc:.2f}%")
      
      self.writer.add_scalar("Top1/train", top1_acc, epoch)
      self.writer.add_scalar("Top5/train", top5_acc, epoch)
      
      if epoch % 100 == 0:
          self.save_checkpoint(epoch)


    def validate_epoch(self, epoch):
        self.model.eval()
        val_losses = []
        correct_1, total_1 = 0, 0
        correct_5, total_5 = 0, 0
        with torch.no_grad():
            for image_ebd, question_ebd, answer in tqdm(self.val_dataloader):
                image_ebd = image_ebd.to(self.device)
                question_ebd = question_ebd.to(self.device)
                answer_label = torch.tensor([self.label_encode(ans) for ans in answer], dtype=torch.long).to(self.device)

                pred = self.model(question_ebd, image_ebd).squeeze(1)
                loss = F.cross_entropy(pred, answer_label)
                val_losses.append(loss.item())

                self.writer.add_scalar("Loss/val", loss.item(), epoch)
                pred = F.softmax(pred, dim=1)

                correct_1_inc, total_1_inc = self.evaluate_accuracy(pred, k=1, answer=answer_label)
                correct_1 += correct_1_inc
                total_1 += total_1_inc

                correct_5_inc, total_5_inc = self.evaluate_accuracy(pred, k=5, answer=answer_label)
                correct_5 += correct_5_inc
                total_5 += total_5_inc

        mean_val_loss = sum(val_losses) / len(val_losses)
        top1_acc = correct_1 / total_1 * 100
        top5_acc = correct_5 / total_5 * 100

        if epoch % 5 == 0:
            print(f"Validation Loss with epoch {epoch}: {mean_val_loss}")
            print(f"Top-{1} Accuracy: {top1_acc:.2f}%")
            print(f"Top-{5} Accuracy: {top5_acc:.2f}%")
        
        self.writer.add_scalar("Top1/val", top1_acc, epoch)
        self.writer.add_scalar("Top5/val", top5_acc, epoch)

        if mean_val_loss < self.best_test_loss:
            self.best_test_loss = mean_val_loss
            self.save_checkpoint(epoch, checkpoint_name="best_model")

    def evaluate_accuracy(self, pred, k, answer):
      _, top_preds = torch.topk(pred, k, dim=1)
      correct = torch.sum(torch.any(top_preds == answer.view(-1, 1), dim=1)).item()
      total = answer.size(0)
      return correct, total

    def configure_optimizer(self):
        learning_rate = self.config.get("learning_rate", 0.00001)
        self.optimizer = Adam(params=self.model.parameters(), lr=learning_rate)

    def get_learning_rate(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def set_learning_rate(self, new_lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def save_checkpoint(self, epoch: int, checkpoint_name='checkpoint'):
        save_checkpoint_path = self.config.get("save_checkpoint_path")
        os.makedirs(save_checkpoint_path, exist_ok=True)
        model_state_dict = self.model.state_dict()
        torch.save(dict(model_state_dict=model_state_dict, epoch=epoch), f"{save_checkpoint_path}/{checkpoint_name}_{epoch:03d}.pkl")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(msg)

    def close_writer(self):
        self.writer.flush()
        self.writer.close()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()