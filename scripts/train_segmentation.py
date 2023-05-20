import os
from math import sqrt
import torchvision.transforms as standard_transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
import utils.joint_transforms as joint_transforms
from models.psp_net import PSPNet
from utils import AverageMeter
from dataset.segmentation_dataset import SegmentationDataset
import PIL
import torchvision


ckpt_path = '../ckpt'
exp_name = 'segmentation'
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def main():
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    if not os.path.exists(os.path.join(ckpt_path, exp_name)):
        os.mkdir(os.path.join(ckpt_path, exp_name))
    writer = SummaryWriter(os.path.join(ckpt_path, exp_name))

    args = {
        'train_batch_size': 2,
        'lr': 1e-2 / sqrt(16 / 2),
        'max_iter': 9e4,
        'snapshot': '',
        'print_freq': 20,
        'val_img_display_size': 512,
        'save_epoch_freq': 10
    }

    net = PSPNet(num_classes=2, pretrained=False)

    if len(args['snapshot']) == 0:
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'iter': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'iter': int(split_snapshot[3]),
                               'val_loss': float(split_snapshot[5]), 'acc': float(split_snapshot[7]),
                               'acc_cls': float(split_snapshot[9]),'mean_iu': float(split_snapshot[11]),
                               'fwavacc': float(split_snapshot[13])}
    net.cuda().train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_shared_transform = joint_transforms.Compose([
        joint_transforms.Scale(512),
        joint_transforms.RandomRotate(50),
        joint_transforms.RandomHorizontallyFlip()

    ])

    train_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        #standard_transforms.Normalize(*mean_std)
    ])

    #train_gt_transform = extended_transforms.MaskToTensor()
    train_gt_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])

    val_shared_transform = joint_transforms.Compose([
        joint_transforms.Scale(512),
    ])

    val_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        #standard_transforms.Normalize(*mean_std)
    ])

    #val_gt_transform = extended_transforms.MaskToTensor()
    val_gt_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*mean_std)
    ])

    train_set = SegmentationDataset('../Data/train_segmentation', train_shared_transform, train_img_transform, train_gt_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True, drop_last=True)
    val_set = SegmentationDataset('../Data/val_segmentation', val_shared_transform, val_img_transform, val_gt_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr']}
    ])

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, writer)


def train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, writer):
    curr_iter = 0
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()

        inputs_all = []
        gts_all = []
        outputs_all = []

        net.train()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr']
            optimizer.param_groups[1]['lr'] = args['lr']
            inputs, gts = data
            inputs, gts = inputs.cuda(), gts.cuda()

            optimizer.zero_grad()
            outputs, aux = net(inputs)

            main_loss = criterion(outputs, gts)
            aux_loss = criterion(aux, gts)
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()

            if i < 10:
                inputs_all.append(inputs)
                outputs_all.append(outputs)
                gts_all.append(gts)

            train_main_loss.update(main_loss.item())
            train_aux_loss.update(aux_loss.item())

            curr_iter += args['train_batch_size']
            writer.add_scalar('train_main_loss', train_main_loss.avg(), curr_iter//inputs.size(0))
            writer.add_scalar('train_aux_loss', train_aux_loss.avg(), curr_iter//inputs.size(0))
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter//inputs.size(0))

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg(), train_aux_loss.avg(),
                    optimizer.param_groups[1]['lr']))
            if curr_iter >= args['max_iter']:
                return

        inputs_all = torch.cat(inputs_all, dim=0)
        gts_all = torch.cat(gts_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)

        outputs_all = torch.argmax(outputs_all, dim=1).float().cpu()
        gts_all = torch.argmax(gts_all, dim=1).float().cpu()

        input_grid = torchvision.utils.make_grid(inputs_all, nrow=len(inputs_all))
        gt_grid = torchvision.utils.make_grid(gts_all, nrow=len(gts_all))
        result_grid = torchvision.utils.make_grid(outputs_all, nrow=len(outputs_all))
        writer.add_image('epoch ' + str(curr_epoch) + ' train_input', input_grid)
        writer.add_image('epoch ' + str(curr_epoch) + ' train_result', result_grid)
        writer.add_image('epoch ' + str(curr_epoch) + ' train_gt', gt_grid)

        validate(val_loader, net, criterion, optimizer, curr_epoch, curr_iter, args, writer)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, iter_num, args, writer):
    # the following code is written assuming that batch size is 1
    net.eval()

    val_loss = AverageMeter()

    inputs_all = []
    gts_all = []
    predictions_all = []

    for vi, data in enumerate(val_loader):
        input, gt = data
        input, gt = input.cuda(), gt.cuda()
        output = net(input)

        val_loss.update(criterion(output, gt).item())

        if (vi + 1) % args['print_freq'] == 0:
            print('validating: %d / %d' % (vi + 1, len(val_loader)))

        # visualization of output image
        input = input[0].cpu()
        inputs_all.append(input)

        output_reformat = output[0]  # BS is 1, so get the element directly
        class_predictions = torch.argmax(output_reformat, dim=0).float().cpu()
        class_predictions = class_predictions.unsqueeze(0)
        predictions_all.append(class_predictions)

        gt = torch.argmax(gt[0], dim=0).float().cpu()
        gt = gt.unsqueeze(0)
        gts_all.append(gt)

    if val_loss.avg() < args['best_record']['val_loss']:
        args['best_record']['val_loss'] = val_loss.avg()
        args['best_record']['epoch'] = epoch
        args['best_record']['iter'] = iter_num
        snapshot_name = 'epoch_%d_iter_%d_loss_%.5f_lr_%.10f' % (
            epoch, iter_num, val_loss.avg(), optimizer.param_groups[1]['lr'])
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    if epoch % args['save_epoch_freq'] == 0:
        snapshot_name = 'epoch_%d_iter_%d_loss_%.5f_lr_%.10f' % (
            epoch, iter_num, val_loss.avg(), optimizer.param_groups[1]['lr'])
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

    input_grid = torchvision.utils.make_grid(inputs_all, nrow=len(val_loader))
    gt_grid = torchvision.utils.make_grid(gts_all, nrow=len(val_loader))
    result_grid = torchvision.utils.make_grid(predictions_all, nrow=len(val_loader))
    writer.add_image('epoch ' + str(epoch) + ' val_input', input_grid)
    writer.add_image('epoch ' + str(epoch) + ' val_result', result_grid)
    writer.add_image('epoch ' + str(epoch) + ' val_gt', gt_grid)

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [iter %d], [val loss %.5f]' % (epoch, iter_num, val_loss.avg()))
    print('best record: [val loss %.5f], [epoch %d], [iter %d]' %
          (args['best_record']['val_loss'], args['best_record']['epoch'], args['best_record']['iter']))
    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg(), epoch)

    return val_loss.avg()


if __name__ == '__main__':
    main()
