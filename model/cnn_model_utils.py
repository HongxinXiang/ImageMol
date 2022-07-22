import logging
import os
import sys
import torch
import torchvision
from sklearn import metrics
from tqdm import tqdm
from model.evaluate import metric as utils_evaluate_metric
from model.evaluate import metric_multitask as utils_evaluate_metric_multitask
from model.evaluate import metric_reg as utils_evaluate_metric_reg
from model.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask


def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    return model


# evaluation for classification
def metric(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    f1 = metrics.f1_score(y_true, y_pred)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_list, precision_list)
    precision = metrics.precision_score(y_true, y_pred, zero_division=1)
    recall = metrics.recall_score(y_true, y_pred, zero_division=1)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    matthews = metrics.matthews_corrcoef(y_true, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob)
    return {
        "accuracy": acc,
        "ROCAUC": auc,
        "f1": f1,
        "AUPR": aupr,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
        "matthews": matthews,
        "fpr": fpr,  # list
        "tpr": tpr,  # list
        "precision_list": precision_list,
        "recall_list": recall_list
    }


def train_one_epoch_multitask(model, optimizer, data_loader, criterion, device, epoch, task_type):
    '''
    :param model:
    :param optimizer:
    :param data_loader:
    :param criterion:
    :param device:
    :param epoch:
    :param criterion_lambda:
    :return:
    '''
    assert task_type in ["classification", "regression"]

    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        pred = model(images)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion(pred.double(), labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification", return_data_dict=False):
    assert task_type in ["classification", "regression"]

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        sample_num += images.shape[0]

        with torch.no_grad():
            pred = model(images)
            labels = labels.view(pred.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            # print(y_true.shape, y_pred.shape, y_pro.shape)
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_pred": y_pred, "y_pro": y_pro}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            if return_data_dict:
                data_dict = {"y_true": y_true, "y_scores": y_scores}
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1]), data_dict
            else:
                return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")


def save_finetune_ckpt(model, optimizer, loss, epoch, save_path, filename_pre, lr_scheduler=None, result_dict=None, logger=None):
    log = logger if logger is not None else logging
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'loss': loss,
            'result_dict': result_dict
        }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log.info("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log.info('model has been saved as {}'.format(filename))

