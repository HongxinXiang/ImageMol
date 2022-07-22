import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from dataloader.JigsawLoader import JigsawDataset, load_pretraining_dataset
from model.feat2image_model import generator, netlocalD
from model.model import ImageMol, Matcher
from model.train_utils import fix_train_random_seed
from utils.public_utils import setup_device


def load_norm_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_tra = [transforms.CenterCrop(args.imageSize),
               transforms.RandomHorizontalFlip(),
               transforms.RandomGrayscale(p=0.2),
               transforms.RandomRotation(degrees=360)]
    tile_tra = [transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor()]
    return normalize, img_tra, tile_tra


def parse_args():
    parser = argparse.ArgumentParser(description='parameters of pretraining ImageMol')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--val_workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', type=int, default=151, help='number of total epochs to run (default: 151)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--checkpoints', type=int, default=1,
                        help='how many iterations between two checkpoints (default: 1)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--dataroot', type=str, default="./datasets/pretraining/", help='data root')
    parser.add_argument('--dataset', type=str, default="toy", help='dataset name, e.g. data, toy')
    parser.add_argument('--ckpt_dir', default='./ckpts/pretrain_model', help='path to checkpoint')
    parser.add_argument('--modelname', type=str, default="ResNet18", choices=["ResNet18"], help='supported model')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--gpu', type=str, default="0", help='GPUs of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--Jigsaw_lambda', type=float, default=1,
                        help='start JPP task, 1 means start, 0 means not start')
    parser.add_argument('--cluster_lambda', type=float, default=1, help='start M3GC task')
    parser.add_argument('--constractive_lambda', type=float, default=0, help='start MCL task')
    parser.add_argument('--matcher_lambda', type=float, default=0, help='start MRD task')
    parser.add_argument('--is_recover_training', type=int, default=1, help='start MIR task')
    parser.add_argument('--cl_mask_type', type=str, default="rectangle_mask", help='',
                        choices=["random_mask", "rectangle_mask", "mix_mask"])
    parser.add_argument('--cl_mask_shape_h', type=int, default=16, help='mask_utils->create_rectangle_mask()')
    parser.add_argument('--cl_mask_shape_w', type=int, default=16, help='mask_utils->create_rectangle_mask()')
    parser.add_argument('--cl_mask_ratio', type=float, default=0.001, help='mask_utils->create_random_mask()')

    return parser.parse_args()


# evaluation
def eval(args, dataloader, model, matcher, netG, netD, criterionBCE, criterion_matcher):
    total = len(dataloader.dataset)

    # evaluation results
    returnData = {
        "JigsawAcc": 0,
        "ClusterAcc100": 0,
        "ClusterAcc1000": 0,
        "ClusterAcc10000": 0,
        "ClusterAcc": 0,
        "ConstractiveLoss": 0,
        "ReasonabilityLoss": 0,
        "RecoverLoss": 0,
        "total": 0
    }

    # evaluation
    with torch.no_grad():
        jigsaw_correct = 0
        class_correct = 0
        class_correct_1 = 0
        class_correct_2 = 0
        class_correct_3 = 0

        for data, jig_l, class_l, data_non_mask, data64_non_mask, cl_data_mask, _ in tqdm(dataloader,
                                                                                          total=len(dataloader)):

            data = data.cuda()
            jig_l = jig_l.cuda()
            class_l1 = class_l[0].cuda()
            class_l2 = class_l[1].cuda()
            class_l3 = class_l[2].cuda()
            data_non_mask = data_non_mask.cuda()
            data64_non_mask = data64_non_mask.cuda()
            cl_data_mask = cl_data_mask.cuda()

            hidden_feat, jig_logit, label1_logit, label2_logit, label3_logit = model(data)

            _, cls_pred1 = label1_logit.max(dim=1)
            _, cls_pred2 = label2_logit.max(dim=1)
            _, cls_pred3 = label3_logit.max(dim=1)

            _, jig_pred = jig_logit.max(dim=1)

            class_correct += torch.sum(cls_pred1 == class_l1.data)
            class_correct += torch.sum(cls_pred2 == class_l2.data)
            class_correct += torch.sum(cls_pred3 == class_l3.data)

            class_correct_1 += torch.sum(cls_pred1 == class_l1.data)
            class_correct_2 += torch.sum(cls_pred2 == class_l2.data)
            class_correct_3 += torch.sum(cls_pred3 == class_l3.data)

            jigsaw_correct += torch.sum(jig_pred == jig_l.data)

            hidden_feat_non_mask, _, _, _, _ = model(data_non_mask)
            hidden_feat_mask, _, _, _, _ = model(cl_data_mask)

            if args.constractive_lambda != 0:
                constractive_loss = (hidden_feat_non_mask - hidden_feat_mask).pow(2).sum(axis=1).sqrt().mean()
                returnData["ConstractiveLoss"] += constractive_loss.item() / total

            if args.matcher_lambda != 0:
                out_cls_false = matcher(hidden_feat)
                out_cls_true = matcher(hidden_feat_non_mask)
                y_out_cls_false = torch.from_numpy(
                    np.where(jig_l.cpu().numpy().copy() > 0, 0, 1)).cuda().long()
                y_out_cls_true = torch.from_numpy(np.ones(out_cls_true.shape[0])).cuda().long()

                reasonability_loss = criterion_matcher(out_cls_false, y_out_cls_false) \
                                     + criterion_matcher(out_cls_true, y_out_cls_true)
                returnData["ReasonabilityLoss"] += reasonability_loss.item() / total

            if args.is_recover_training == 1:
                real_label = 1
                fake_label = 0
                ################### train D ###################
                netD.zero_grad()
                label = torch.FloatTensor(data64_non_mask.shape[0]).cuda()
                label.data.resize_(data64_non_mask.shape[0]).fill_(real_label)
                output = netD(data64_non_mask)
                errD_real = criterionBCE(output.flatten(), label)
                # train with fake
                hidden_feat_crop, _, _, _, _ = model(data)
                fake = netG(hidden_feat_crop)
                label.data.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterionBCE(output.flatten(), label)
                errD = errD_real + errD_fake
                ################### train G ###################
                netG.zero_grad()
                label.data.fill_(real_label)  # fake labels are real for generator cost
                output = netD(fake)
                errG_D = criterionBCE(output.flatten(), label)
                errG_l2 = (fake - data64_non_mask).pow(2)
                errG_l2 = errG_l2.mean()
                errG = (errG_D + errG_l2)

                returnData["RecoverLoss"] += (errD.item() + errG.item()) / 2 / total

        jigsaw_acc = float(jigsaw_correct) / total
        class_acc = float(class_correct) / (total * 3)

    returnData["JigsawAcc"] = jigsaw_acc
    returnData["ClusterAcc100"] = float(class_correct_1) / total
    returnData["ClusterAcc1000"] = float(class_correct_2) / total
    returnData["ClusterAcc10000"] = float(class_correct_3) / total
    returnData["ClusterAcc"] = class_acc
    returnData["total"] = total

    return returnData


def main(args):
    start_time = datetime.now()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.seed)

    # default params
    jigsaw_classes = 100 + 1
    label1_classes = 100
    label2_classes = 1000
    label3_classes = 10000
    val_size = 0.05
    original_image_rate = 0.8
    eval_each_batch = 1000

    # load model
    model = ImageMol(args.modelname, jigsaw_classes, label1_classes=label1_classes, label2_classes=label2_classes,
                     label3_classes=label3_classes)
    matcher = Matcher()
    netG = generator(input_dim=512)
    netD = netlocalD(args)

    print(model)
    print(matcher)
    print(netG)
    print(netD)

    if len(device_ids) > 1:
        print("starting multi-gpu.")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        matcher = torch.nn.DataParallel(matcher, device_ids=device_ids)
        netG = torch.nn.DataParallel(netG, device_ids=device_ids)
        netD = torch.nn.DataParallel(netD, device_ids=device_ids)

    model = model.cuda()
    matcher = matcher.cuda()
    netG = netG.cuda()
    netD = netD.cuda()

    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )
    optimizerM = torch.optim.Adam(matcher.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_matcher = torch.nn.NLLLoss().cuda()
    criterionBCE = torch.nn.BCELoss().cuda()

    # load data
    normalize, img_tra, tile_tra = load_norm_transform()

    name_train, name_val, labels_train, labels_val = load_pretraining_dataset(args.dataroot, args.dataset, val_size)
    train_dataset = JigsawDataset(name_train, labels_train, img_transformer=transforms.Compose(img_tra),
                                  tile_transformer=transforms.Compose(tile_tra),
                                  bias_whole_image=original_image_rate,
                                  normalize=normalize,
                                  args=args)
    val_dataset = JigsawDataset(name_val, labels_val, img_transformer=transforms.Compose(img_tra),
                                tile_transformer=transforms.Compose(tile_tra),
                                bias_whole_image=original_image_rate,
                                normalize=normalize,
                                args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.val_workers,
                                                 # sampler=sampler,
                                                 pin_memory=True)

    # starting to train
    for epoch in range(args.start_epoch, args.epochs):

        # switch to train mode
        model.train()

        AvgConstractiveLoss = 0
        AvgReasonabilityLoss = 0
        AvgRecoverLoss = 0
        AvgJigLoss = 0
        AvgClassLoss1 = 0
        AvgClassLoss2 = 0
        AvgClassLoss3 = 0
        AvgClassLoss = 0
        AvgTotalLoss = 0
        with tqdm(total=len(train_dataloader)) as t:
            for i, (
                    Jigsaw_img, Jigsaw_label, original_label, data_non_mask, data64_non_mask, cl_data_mask,
                    _) in enumerate(
                train_dataloader):

                Jigsaw_img_var = torch.autograd.Variable(Jigsaw_img.cuda())
                Jigsaw_label_var = torch.autograd.Variable(Jigsaw_label.cuda())
                data_non_mask = torch.autograd.Variable(data_non_mask.cuda())
                data64_non_mask = torch.autograd.Variable(data64_non_mask.cuda())
                cl_data_mask = torch.autograd.Variable(cl_data_mask.cuda())

                original_label1_var = torch.autograd.Variable(original_label[0].cuda())
                original_label2_var = torch.autograd.Variable(original_label[1].cuda())
                original_label3_var = torch.autograd.Variable(original_label[2].cuda())

                hidden_feat, pre_Jigsaw_label, pre_class_label1, pre_class_label2, pre_class_label3 = model(
                    Jigsaw_img_var)

                Jig_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.Jigsaw_lambda != 0:
                    Jig_loss = criterion(pre_Jigsaw_label, Jigsaw_label_var)

                class_loss1 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                class_loss2 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                class_loss3 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                class_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.cluster_lambda != 0:
                    class_loss1 = criterion(pre_class_label1, original_label1_var)
                    class_loss2 = criterion(pre_class_label2, original_label2_var)
                    class_loss3 = criterion(pre_class_label3, original_label3_var)
                    class_loss = class_loss1 + class_loss2 + class_loss3

                hidden_feat_non_mask, _, _, _, _ = model(data_non_mask)
                hidden_feat_mask, _, _, _, _ = model(cl_data_mask)
                constractive_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.constractive_lambda != 0:
                    constractive_loss = (hidden_feat_non_mask - hidden_feat_mask).pow(2).sum(axis=1).sqrt().mean()
                    AvgConstractiveLoss += constractive_loss.item() / len(train_dataloader)

                reasonability_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.matcher_lambda != 0:
                    out_cls_false = matcher(hidden_feat)
                    out_cls_true = matcher(hidden_feat_non_mask)
                    y_out_cls_false = torch.from_numpy(
                        np.where(Jigsaw_label.numpy().copy() > 0, 0, 1)).cuda().long()
                    y_out_cls_true = torch.from_numpy(np.ones(out_cls_true.shape[0])).cuda().long()

                    reasonability_loss = criterion_matcher(out_cls_false, y_out_cls_false) \
                                         + criterion_matcher(out_cls_true, y_out_cls_true)
                    AvgReasonabilityLoss += reasonability_loss.item() / len(train_dataloader)

                errG = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                errD = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
                if args.is_recover_training == 1:
                    real_label = 1
                    fake_label = 0
                    ################### train D ###################
                    netD.zero_grad()
                    label = torch.FloatTensor(data64_non_mask.shape[0]).cuda()
                    label.data.resize_(data64_non_mask.shape[0]).fill_(real_label)

                    output = netD(data64_non_mask)
                    errD_real = criterionBCE(output.flatten(), label)
                    errD_real.backward()

                    # train with fake
                    hidden_feat_crop, _, _, _, _ = model(Jigsaw_img_var)
                    fake = netG(hidden_feat_crop)
                    label.data.fill_(fake_label)
                    output = netD(fake.detach())
                    errD_fake = criterionBCE(output.flatten(), label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    optimizerD.step()
                    optimizerD.zero_grad()

                    ################### train G ###################
                    netG.zero_grad()
                    label.data.fill_(real_label)
                    output = netD(fake)
                    errG_D = criterionBCE(output.flatten(), label)
                    errG_l2 = (fake - data64_non_mask).pow(2)
                    errG_l2 = errG_l2.mean()
                    errG = (errG_D + errG_l2)
                    errG.backward()
                    errG = errG
                    optimizerG.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizerG.zero_grad()

                    AvgRecoverLoss += (errD.item() + errG.item()) / 2 / len(train_dataloader)

                # calculating all loss to backward
                loss = class_loss * args.cluster_lambda + args.Jigsaw_lambda * Jig_loss + args.constractive_lambda * constractive_loss + args.matcher_lambda * reasonability_loss

                # calculating average loss
                AvgJigLoss += Jig_loss.item() / len(train_dataloader)
                AvgClassLoss1 += class_loss1.item() / len(train_dataloader)
                AvgClassLoss2 += class_loss2.item() / len(train_dataloader)
                AvgClassLoss3 += class_loss3.item() / len(train_dataloader)
                AvgClassLoss += class_loss.item() / len(train_dataloader)
                AvgTotalLoss += loss.item() / len(train_dataloader)

                # compute gradient and do SGD step
                if loss.item() != 0:
                    loss.backward()
                    optimizer.step()
                    optimizerM.step()
                    optimizer.zero_grad()
                    optimizerM.zero_grad()

                if args.verbose and (i % eval_each_batch) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'TotalLoss: {}\t'
                          'ClsTotalLoss: {}\t'
                          'ClsLoss_100: {}\t'
                          'ClsLoss_1000: {}\t'
                          'ClsLoss_10000: {}\t'
                          'C_Loss: {}\t'
                          'JigLoss: {}\t'
                          'M_Loss: {}\t'
                          'errD: {}\t'
                          'errG: {}\t'
                          'mean(errD+errG): {}\t'
                          .format(epoch + 1, i, len(train_dataloader), loss.item(), class_loss.item(),
                                  class_loss1.item(), class_loss2.item(), class_loss3.item(),
                                  constractive_loss.item(),
                                  Jig_loss.item(), reasonability_loss.item(), errD.item(), errG.item(),
                                  (errD.item() + errG.item()) / 2))

                t.set_postfix(TotalLoss=loss.item(), ClsTotalLoss=class_loss.item(), ClsLoss_100=class_loss1.item(),
                              ClsLoss_1000=class_loss2.item(), ClsLoss_10000=class_loss3.item(),
                              C_loss=constractive_loss.item(), JigLoss=Jig_loss.item(),
                              M_loss=reasonability_loss.item(),
                              errD=errD.item(), errG=errG.item())
                t.update(1)

        # evaluation
        model.eval()
        evaluationData = eval(args, val_dataloader, model, matcher, netG, netD, criterionBCE, criterion_matcher)

        # save model
        saveRoot = os.path.join(args.ckpt_dir, 'checkpoints')
        if not os.path.exists(saveRoot):
            os.makedirs(saveRoot)
        if epoch % args.checkpoints == 0:
            saveFile = os.path.join(saveRoot, 'ImageMol_{}.pth.tar'.format(epoch + 1))
            if args.verbose:
                print('Save checkpoint at: {}'.format(saveFile))

            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'arch': args.modelname,
                'state_dict': model_state_dict,
            }, saveFile)

        print('Epoch: [{}][train]\t'
              'TotalLoss: {}\t'
              'JigLoss: {}\t'
              'ClsLoss_100: {}\t'
              'ClsLoss_1000: {}\t'
              'ClsLoss_10000: {}\t'
              'ClsTotalLoss(fftotal): {}\t'
              'AvgConstractiveLoss: {}\t'
              'AvgReasonabilityLoss: {}\t'
              'AvgRecoverLoss: {}\t'
              .format(epoch + 1, AvgTotalLoss, AvgJigLoss, AvgClassLoss1,
                      AvgClassLoss2, AvgClassLoss3, AvgClassLoss,
                      AvgConstractiveLoss, AvgReasonabilityLoss, AvgRecoverLoss))

        print('Epoch: [{}][val]\t'
              'JigsawAcc: {}\t'
              'ClusterAcc100: {}\t'
              'ClusterAcc1000: {}\t'
              'ClusterAcc10000: {}\t'
              'ClusterAcc(avg): {}\t'
              'ConstractiveLoss: {}\t'
              'ReasonabilityLoss: {}\t'
              'RecoverLoss: {}\t'
              .format(epoch + 1, evaluationData["JigsawAcc"], evaluationData["ClusterAcc100"],
                      evaluationData["ClusterAcc1000"], evaluationData["ClusterAcc10000"],
                      evaluationData["ClusterAcc"], evaluationData["ConstractiveLoss"],
                      evaluationData["ReasonabilityLoss"], evaluationData["RecoverLoss"]))

    end_time = datetime.now()
    print("used time: {}".format(end_time - start_time))


if __name__ == '__main__':
    args = parse_args()
    main(args)
