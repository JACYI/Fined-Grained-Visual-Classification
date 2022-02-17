import os
import time
import glob
import torch
import logging
import argparse
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import cv2
import torchvision.transforms as transforms
import shutil
# from requests.utils import urlparse
# import wget

import model
import model.resnet50
import model.vgg19
from utils.utils import load_config, setup_seed, process_bar
# from utils.visualize import Visualizer
from utils.transform import UnNormalizer
from PIL import Image
import tqdm


def main():
    model_options = ['resnet50', 'vgg19']

    parser = argparse.ArgumentParser(description='MM-CNN')
    # parser.add_argument('--dataset', '-d', default='birds',
    #                     choices=dataset_options)
    parser.add_argument('--model', '-a', default='resnet50',
                        choices=model_options)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument("--gpu", type=int, default=0,
                        help='gpu index (default: 0)')
    # parser.add_argument('--visualize', action='store_true', default=False,
    #                     help='plot attention masks and ROIs')

    args = parser.parse_args()

    #####################
    #  set the dataset  #
    #####################
    args.dataset = 'rs'

    setup_seed(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    ### prepare configurations
    config_file = "configs/config_{}.yaml".format(args.dataset)
    config = load_config(config_file)
    # data config
    train_dir = config['train_dir']
    test_dir = config['test_dir']
    num_class = config['num_class']
    # model config
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    weight_decay = float(config['weight_decay'])
    num_epoch = config['num_epoch']
    resize_size = config['resize_size']
    crop_size = config['crop_size']
    # visualizer config
    # vis_host = config['vis_host']
    # vis_port = config['vis_port']

    ################
    #  model path  #
    ################
    model_pth = "/home/yiyonghao/IGARSS/MM-CNN/logs/baseline/AP-CNN_resnet50_rs_11-20-17-40/model_best.pth"

    ################
    #  image path  #
    ################
    test_dir = '/home/yiyonghao/IGARSS/dataset/big_aircraft/test'
    input_image_dir = config['test_dir']
    image_format = '.jpg'
    output_dir = '/home/yiyonghao/IGARSS/dataset/big_aircraft/'
    # output_dir = os.path.dirname(model_pth)
    roi_dir = os.path.join(output_dir, 'vision')
    if os.path.exists(roi_dir):
        os.makedirs(roi_dir, exist_ok=True)
    # mask_dir = os.path.join(output_dir, 'mask')
    # if os.path.exists(mask_dir):
    #     os.makedirs(mask_dir, exist_ok=True)
    # output roi and attention
    visualize = False


    ### setup exp_dir
    # exp_name = "AP-CNN_{}_{}".format(args.model, args.dataset)
    # time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    # exp_dir = os.path.join("./logs/test", exp_name + '_' + time_str)
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)
    # generate log files
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO, filemode='w')
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(levelname)-4s %(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)

    # logging.info('==>exp dir:%s' % exp_dir)
    # logging.info("OPENING " + exp_dir + '/results_train.csv')
    # logging.info("OPENING " + exp_dir + '/results_test.csv')
    #
    # results_train_file = open(exp_dir + '/results_train.csv', 'w')
    # results_train_file.write('epoch, train_acc, train_loss\n')
    # results_train_file.flush()
    # results_test_file = open(exp_dir + '/results_test.csv', 'w')
    # results_test_file.write('epoch, test_acc, test_loss\n')
    # results_test_file.flush()

    # set up Visualizer
    # vis = Visualizer(env=exp_name, port=vis_port, server=vis_host)

    ### preparing data
    print('Preparing data..')

    transform_train = transforms.Compose([
        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    unorm = UnNormalizer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    ################
    #  load image  #
    ################
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Successfully Preparing data..')

    # image_path_list = glob.glob(os.path.join(
    #     input_image_dir, '*{}'.format(image_format)))
    # if image_path_list == []:
    #     print('no image input')
    #     return
    # print('inference list', os.path.join(
    #     input_image_dir, '*{}'.format(image_format)))

    ### building model
    # logging.info('==> Building model..')
    # load pretrained backbone on ImageNet
    # pretrained_path = ""
    # if args.model == "resnet50":
    #     # url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    #     pretrained_path = "/home/yiyonghao/.torch/models/resnet50-19c8e357.pth"
    # elif args.model == "vgg19":
    #     # url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
    #     pretrained_path = "/home/yiyonghao/.torch/models/vgg19_bn-c79401a0.pth"
    # model_dir = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch/models'))
    # filename = os.path.basename(urlparse(url).path)
    # pretrained_path = os.path.join(model_dir, filename)
    # print(pretrained_path)

    # if not os.path.exists(pretrained_path):
    #     wget.download(url, pretrained_path)
    net = getattr(getattr(model, args.model), args.model)(num_class)
    if model_pth:
        print('load trained model')
        # net_dict = net.state_dict()
        trained_dict = torch.load(model_pth)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        # net_dict.update(pretrained_dict)
        net.load_state_dict(trained_dict)
    else:
        print('model path not found!')
        return
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        # if torch.cuda.device_count() > 1:
        #     net = nn.DataParallel(net)
    print('Successfully Building model..')

    ### test scripts
    def test():
        net.eval()
        with torch.no_grad():
            # total_correct_num = 0
            # average_accuracy = 0
            # NUM_LABLE_DICT = {}
            # CORRECT_NUM_LABLE_DICT = {}
            # index_label_map = test_dataset.index_label_map
            #
            # for image_path in tqdm(image_path_list):
            #     src_image = Image.open(image_path).convert('RGB')
            #     infer_transform = get_transform(resize=RESIZE, phase='test')
            #     image = infer_transform(src_image)
            #     image_name = image_path.split(image_format)[0].split('/')[-1]
            #     label = image_name.split('_')[-1]
            #
            #     X = image.to(device).unsqueeze(0)

            test_loss = 0
            correct = 0
            total = 0
            idx = 0
            count = 0
            cls_numbers_dict = {}
            cls_correct_dict = {}
            test_acc = 0
            # lebel_index to classes name map
            idx_cls = testset.classes

            for batch_idx, (inputs, targets) in enumerate(testloader):# batch loop
                # process bar print
                batchs = int(len(testset.imgs)/batch_size)
                process_bar(float(batch_idx)/batchs, batch_idx, ' ', str(batchs), 50)

                idx = batch_idx
                gt_label_list = targets.numpy()
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                loss_ret, acc_ret, predicted = net(inputs, targets)
                # roi_3, roi_4, roi_5 = roi_list
                pt_label_list = predicted.data.cpu().numpy()
                # statistic the number for single class
                for i, gt_label in enumerate(gt_label_list): # single image loop
                    pt_label = pt_label_list[i]

                    # predicted equals groundtruth, correct number update
                    if pt_label == gt_label:
                        if gt_label not in cls_correct_dict:
                            cls_correct_dict[gt_label] = 1
                        else:
                            cls_correct_dict[gt_label] += 1
                    # total numbers update
                    if gt_label not in cls_numbers_dict:
                        cls_numbers_dict[gt_label] = 1
                    else:
                        cls_numbers_dict[gt_label] += 1


                    # plot and save the roi boundary box on images
                    if pt_label == gt_label:
                        roi_dst_dir = os.path.join(roi_dir, 'correct')
                        # mask_dst_dir = os.path.join(mask_dir, 'correct')
                    else:
                        roi_dst_dir = os.path.join(roi_dir, 'error')
                        # mask_dst_dir = os.path.join(mask_dir, 'error')
                    roi_dst_dir = os.path.join(roi_dst_dir, idx_cls[int(gt_label)])
                    # mask_dst_dir = os.path.join(mask_dst_dir, idx_cls[int(gt_label)])
                    if not os.path.exists(roi_dst_dir):
                        os.makedirs(roi_dst_dir)
                    # if not os.path.exists(mask_dst_dir):
                    #     os.makedirs(mask_dst_dir)
                    roi_file_path = os.path.join(roi_dst_dir,
                                            idx_cls[int(pt_label)] + '_' + \
                                             testset.imgs[batch_size*batch_idx + i][0].split('/')[-1].split('.')[0] + '.jpg')
                    img = unorm(inputs[i].cpu()).numpy().copy()
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])
                    # change the channel(BGR->RGB)
                    r, g, b = cv2.split(img)
                    img = cv2.merge([b, g, r])
                    cv2.imwrite(roi_file_path, img)
                    # mask_file_path = os.path.join(mask_dst_dir,
                    #                         idx_cls[int(pt_label)] + '_' + \
                    #                          testset.imgs[batch_size*batch_idx + i][0].split('/')[-1] + '_mask' + '.jpg')

                    # if visualize:
                    #     # roi
                    #     single_roi_3 = roi_3[roi_3[:, 0] == i]
                    #     single_roi_4 = roi_4[roi_4[:, 0] == i]
                    #     single_roi_5 = roi_5[roi_5[:, 0] == i]
                    #     single_roi_list = [single_roi_3, single_roi_4, single_roi_5]
                    #     plot_single_image_roi(inputs[i], single_roi_list, unorm, vis=None, mode='test', file_path=roi_file_path)
                    #
                    #     # mask
                    #     plot_single_image_mask(inputs[i], mask_cat[i], unorm, vis=None, mode='test', file_path=mask_file_path)


                loss = loss_ret['loss']
                test_loss += loss.data
                total += targets.size(0)
                correct += acc_ret['acc']
                # if visualize:
                #     plot_mask_cat(inputs, mask_cat, unorm, None,
                #                   dst_dir='/home/yiyonghao/git/AP-CNN_Pytorch-master/output/error_test/airs/mask',
                #                   label_lists=label_lists,
                #                   idx_cls=idx_cls)
                    # plot_roi(inputs, roi_list, unorm, None,
                    #          dir='/home/yiyonghao/git/AP-CNN_Pytorch-master/output/error_test/airs',
                    #          label_lists=label_lists,
                    #          idx_cls=idx_cls)
        print('\n')
        test_acc = 100. * correct / total
        test_loss = test_loss / (idx + 1)

        # results_test_file.flush()
        return test_acc, test_loss, cls_correct_dict, cls_numbers_dict


    test_acc, test, cls_correct_dict, cls_numbers_dict = test()
    cls_acc_list = {}
    total_number = 0
    total_correct = 0
    for (cls_idx, cls_number) in cls_numbers_dict.items():
        if cls_number == 0:
            cls_acc_list[cls_idx] = 0.
        else:
            correct = 0 if cls_idx not in cls_correct_dict.keys() else cls_correct_dict[cls_idx]
            cls_acc_list[cls_idx] = float(correct)/cls_number
            total_number += cls_number
            total_correct += correct
    txt = os.path.join(os.path.dirname(roi_dir), 'acc_cls.txt')

    with open(txt, 'w') as txt_file:
        for (cls_idx, cls_acc) in cls_acc_list.items():
            txt_file.write('[class:{}]{:.4f}\n'.format(testset.classes[cls_idx], cls_acc))
    print('txt_file wrote!')
    print('number:{}\taccuracy:{:.4f}'.format(total_number, float(total_correct)/total_number))
    # if test_acc > max_test_acc:
    #     max_test_acc = test_acc
    #     torch.save(net.state_dict(), os.path.join(exp_dir, 'model_best.pth'))



if __name__ == "__main__":
    main()