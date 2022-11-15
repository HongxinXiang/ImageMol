import argparse
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # basic
    parser.add_argument('--dataset', type=str, default="BBBP", help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced'],
                        help='regularization of classification loss')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--weighted_CE', action='store_true', default=False, help='whether to use global imbalanced weight')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')

    return parser.parse_args()


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    args.verbose = True

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # architecture name
    if args.verbose:
        print('Architecture: {}'.format(args.image_model))

    ##################################### initialize some parameters #####################################
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    ##################################### load data #####################################
    if args.image_aug:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                 transforms.ToTensor()]
    else:
        img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    num_tasks = labels.shape[1]

    if args.split == "random":
        train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, len(names))), frac_train=0.8,
                                                                frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split == "stratified":
        train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(list(range(0, len(names))), labels,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=args.seed)
    elif args.split == "scaffold":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, len(names))), smiles, frac_train=0.8,
                                                                     frac_valid=0.1, frac_test=0.1)
    elif args.split == "random_scaffold":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(list(range(0, len(names))), smiles,
                                                                            frac_train=0.8, frac_valid=0.1,
                                                                            frac_test=0.1, seed=args.seed)
    elif args.split == "scaffold_balanced":
        smiles = load_smiles(args.txt_file)
        train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(list(range(0, len(names))), smiles,
                                                                              frac_train=0.8, frac_valid=0.1,
                                                                              frac_test=0.1, seed=args.seed,
                                                                              balanced=True)

    name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[
        test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize, args=args)
    val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args)
    test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### load model #####################################
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks)

    if args.resume:
        if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            ckp_keys = list(checkpoint['state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            if args.image_model == "ResNet18":
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

            model.load_state_dict(model_sd)
            arch = checkpoint['arch']
            print("resume model info: arch: {}".format(arch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print(model)
    print("params: {}".format(cal_torch_model_params(model)))
    model = model.cuda()
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    ##################################### initialize optimizer #####################################
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.weight_decay,
    )

    weights = None
    if args.task_type == "classification":
        if args.weighted_CE:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()}
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    ##################################### train #####################################
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                                  weights=weights, device=device, epoch=epoch, task_type=args.task_type)
        # evaluate
        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                                           criterion=criterion, device=device,
                                                                           epoch=epoch, task_type=args.task_type,
                                                                           return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                                     criterion=criterion, device=device,
                                                                     epoch=epoch, task_type=args.task_type,
                                                                     return_data_dict=True)
        test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                                        criterion=criterion, device=device, epoch=epoch,
                                                                        task_type=args.task_type, return_data_dict=True)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            results['final_test_desc'] = test_results

            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(train_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print("final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}"
          .format(results["highest_valid"], results["final_train"], results["final_test"]))


if __name__ == "__main__":
    args = parse_args()
    main(args)
