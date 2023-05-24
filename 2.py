import os
import shutil
import cv2
import numpy as np
import re
import json
import paddle
from paddle.io import Dataset, DataLoader
from Class3.data_transform import Compose, Normalize, RandomSacle, RandomFlip,ConvertDataType,Resize
import matplotlib.pyplot as plt
import paddle.nn as nn
from paddle.nn import functional as F
from PIL import Image

def moveImgDir(color_dir, newcolor_dir, mask_dir, newmask_dir, frames_dir, newframes_dir):
    filenames = os.listdir(color_dir)
    filenames.sort()
    for index, filename in enumerate(filenames):
        src = os.path.join(color_dir,filename)
        dst = os.path.join(newcolor_dir,filename)
        shutil.move(src, dst)
        # colors 文件夹中的文件名多了GT，所以要去掉
        new_filename = re.sub('GT', '', filename)
        src = os.path.join(mask_dir, new_filename)
        dst = os.path.join(newmask_dir, new_filename)
        shutil.move(src, dst)
        src = os.path.join(frames_dir, new_filename)
        dst = os.path.join(newframes_dir, new_filename)
        shutil.move(src, dst)
        if index == 50:
            break

moveImgDir(r"work/dataset/colors", r"work/dataset/val_colors",
r"work/dataset/masks", r"work/dataset/val_masks",
r"work/dataset/frames", r"work/dataset/val_frames")

labels = ['Background', 'Asphalt', 'Paved', 'Unpaved',
          'Markings', 'Speed-Bump', 'Cats-Eye', 'Storm-Drain',
          'Patch', 'Water-Puddle', 'Pothole', 'Cracks']

label_color_dict = {}
mask_dir = r"work/dataset/masks"
color_dir = r"work/dataset/colors"
mask_names = [f for f in os.listdir(mask_dir) if f.endswith('png')]
color_names = [f for f in os.listdir(color_dir) if f.endswith('png')]

for index, label in enumerate(labels):
    if index >= 8:
        index += 1

    for color_name in color_names:
        color = cv2.imread(os.path.join(color_dir, color_name), -1)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        mask_name = re.sub('GT', '', color_name)
        mask = cv2.imread(os.path.join(mask_dir, mask_name), -1)
        mask_color = color[np.where(mask == index)]
        if len(mask_color) != 0:
            label_color_dict[label] = list(mask_color[0].astype(float))
            break

with open(r"work/dataset/mask2color.json", "w", encoding='utf-8') as f:
    # json.dump(dict_, f)  # 写为一行
    json.dump(label_color_dict, f, indent=2, sort_keys=True, ensure_ascii=False)

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


class BasicDataset(Dataset):
    '''
    需要读取数据并返回转换过的数据以及标签，数据和标签的后缀均为.png
    '''

    def __init__(self, image_folder, label_folder, size):
        super(BasicDataset, self).__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.path_Img = get_paths_from_images(image_folder)
        if label_folder is not None:
            self.path_Label = get_paths_from_images(label_folder)
        self.size = size
        self.transform = Compose(
            [RandomSacle(),
             RandomFlip(),
             Resize(self.size),
             ConvertDataType(),
             Normalize(0, 1)
             ]
        )

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h == h_gt, "error"
        assert w == w_gt, "error"
        data, label = self.transform(data, label)
        label = label[:, :, np.newaxis]
        return data, label

    def __getitem__(self, index):
        Img_path, Label_path = None, None
        Img_path = self.path_Img[index]
        Label_path = self.path_Label[index]

        data = cv2.imread(Img_path, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = cv2.imread(Label_path, cv2.IMREAD_GRAYSCALE)
        data, label = self.preprocess(data, label)
        return {'Image': data, 'Label': label}

    def __len__(self):
        return len(self.path_Img)


class Basic_ValDataset(Dataset):
    '''
    需要读取数据并返回转换过的数据、标签以及图像数据的路径
    '''

    def __init__(self, image_folder, label_folder, size):
        super(Basic_ValDataset, self).__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.path_Img = get_paths_from_images(image_folder)
        if label_folder is not None:
            self.path_Label = get_paths_from_images(label_folder)
        self.size = size
        self.transform = Compose(
            [Resize(size),
             ConvertDataType(),
             Normalize(0, 1)
             ]
        )

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h == h_gt, "error"
        assert w == w_gt, "error"
        data, label = self.transform(data, label)
        label = label[:, :, np.newaxis]
        return data, label

    def __getitem__(self, index):
        Img_path, Label_path = None, None
        Img_path = self.path_Img[index]
        Label_path = self.path_Label[index]

        data = cv2.imread(Img_path, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = cv2.imread(Label_path, cv2.IMREAD_GRAYSCALE)
        data, label = self.preprocess(data, label)
        return {'Image': data, 'Label': label, 'Path': Img_path}

    def __len__(self):
        return len(self.path_Img)


class SeparableConv2D(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(SeparableConv2D, self).__init__()

        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._in_channels = in_channels
        self._data_format = data_format

        # 第一次卷积参数，没有偏置参数
        filter_shape = [in_channels, 1] + self.convert_to_list(kernel_size, 2, 'kernel_size')
        self.weight_conv = self.create_parameter(shape=filter_shape, attr=weight_attr)

        # 第二次卷积参数
        filter_shape = [out_channels, in_channels] + self.convert_to_list(1, 2, 'kernel_size')
        self.weight_pointwise = self.create_parameter(shape=filter_shape, attr=weight_attr)
        self.bias_pointwise = self.create_parameter(shape=[out_channels],
                                                    attr=bias_attr,
                                                    is_bias=True)

    def convert_to_list(self, value, n, name, dtype=np.int):
        if isinstance(value, dtype):
            return [value, ] * n
        else:
            try:
                value_list = list(value)
            except TypeError:
                raise ValueError("The " + name +
                                 "'s type must be list or tuple. Received: " + str(
                    value))
            if len(value_list) != n:
                raise ValueError("The " + name + "'s length must be " + str(n) +
                                 ". Received: " + str(value))
            for single_value in value_list:
                try:
                    dtype(single_value)
                except (ValueError, TypeError):
                    raise ValueError(
                        "The " + name + "'s type must be a list or tuple of " + str(
                            n) + " " + str(dtype) + " . Received: " + str(
                            value) + " "
                                     "including element " + str(single_value) + " of type" + " "
                        + str(type(single_value)))
            return value_list

    def forward(self, inputs):
        conv_out = F.conv2d(inputs,
                            self.weight_conv,
                            padding=self._padding,
                            stride=self._stride,
                            dilation=self._dilation,
                            groups=self._in_channels,
                            data_format=self._data_format)
        out = F.conv2d(conv_out,
                       self.weight_pointwise,
                       bias=self.bias_pointwise,
                       padding=0,
                       stride=1,
                       dilation=1,
                       groups=1,
                       data_format=self._data_format)
        print('out:{}'.format(out.shape))
        return out


class Encoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.relus = paddle.nn.LayerList(
            [paddle.nn.ReLU() for i in range(2)])
        self.separable_conv_01 = SeparableConv2D(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.bns = paddle.nn.LayerList(
            [paddle.nn.BatchNorm2D(out_channels) for i in range(2)])

        self.separable_conv_02 = SeparableConv2D(out_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.residual_conv = paddle.nn.Conv2D(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=2,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relus[0](inputs)
        y = self.separable_conv_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.separable_conv_02(y)
        y = self.bns[1](y)
        y = self.pool(y)

        residual = self.residual_conv(previous_block_activation)
        y = paddle.add(y, residual)

        return y


class Decoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.relus = paddle.nn.LayerList(
            [paddle.nn.ReLU() for i in range(2)])
        self.conv_transpose_01 = paddle.nn.Conv2DTranspose(in_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding=1)
        self.conv_transpose_02 = paddle.nn.Conv2DTranspose(out_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding=1)
        self.bns = paddle.nn.LayerList(
            [paddle.nn.BatchNorm2D(out_channels) for i in range(2)]
        )
        self.upsamples = paddle.nn.LayerList(
            [paddle.nn.Upsample(scale_factor=2.0) for i in range(2)]
        )
        self.residual_conv = paddle.nn.Conv2D(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relus[0](inputs)
        y = self.conv_transpose_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.conv_transpose_02(y)
        y = self.bns[1](y)
        y = self.upsamples[0](y)

        residual = self.upsamples[1](previous_block_activation)
        residual = self.residual_conv(residual)

        y = paddle.add(y, residual)

        return y


class UNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.conv_1 = paddle.nn.Conv2D(3, 32,
                                       kernel_size=3,
                                       stride=2,
                                       padding='same')
        self.bn = paddle.nn.BatchNorm2D(32)
        self.relu = paddle.nn.ReLU()

        in_channels = 32
        self.encoders = []
        self.encoder_list = [64, 128, 256]
        self.decoder_list = [256, 128, 64, 32]

        # 根据下采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.encoder_list:
            block = self.add_sublayer('encoder_{}'.format(out_channels),
                                      Encoder(in_channels, out_channels))
            self.encoders.append(block)
            in_channels = out_channels

        self.decoders = []

        # 根据上采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.decoder_list:
            block = self.add_sublayer('decoder_{}'.format(out_channels),
                                      Decoder(in_channels, out_channels))
            self.decoders.append(block)
            in_channels = out_channels

        self.output_conv = paddle.nn.Conv2D(in_channels,
                                            num_classes,
                                            kernel_size=3,
                                            padding='same')

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)

        for encoder in self.encoders:
            y = encoder(y)

        for decoder in self.decoders:
            y = decoder(y)

        y = self.output_conv(y)
        return y


class PSPNet(nn.Layer):
    def __init__(self, num_classes=13, backbone_name="resnet50"):
        super(PSPNet, self).__init__()

        if backbone_name == "resnet50":
            backbone = resnet50(pretrained=True)
        if backbone_name == "resnet101":
            backbone = resnet101(pretrained=True)

        # self.layer0 = nn.Sequential(*[backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool])
        self.layer0 = nn.Sequential(*[backbone.conv1, backbone.maxpool])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        num_channels = 2048

        self.pspmodule = PSPModule(num_channels, [1, 2, 3, 6])

        num_channels *= 2

        self.classifier = nn.Sequential(*[
            nn.Conv2D(num_channels, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Dropout2D(0.1),
            nn.Conv2D(512, num_classes, 1)
        ])

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.classifier(x)
        out = F.interpolate(
            x,
            paddle.shape(inputs)[2:],
            mode='bilinear',
            align_corners=True)

        return out


class PSPModule(nn.Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters = num_channels // len(bin_size_list)
        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(nn.Sequential(*[
                nn.Conv2D(num_channels, num_filters, 1),
                nn.BatchNorm2D(num_filters),
                nn.ReLU()
            ]))

    def forward(self, inputs):
        out = [inputs]
        for idx, f in enumerate(self.features):
            pool = paddle.nn.AdaptiveAvgPool2D(self.bin_size_list[idx])
            x = pool(inputs)
            x = f(x)
            x = F.interpolate(x, paddle.shape(inputs)[2:], mode="bilinear", align_corners=True)
            out.append(x)

        out = paddle.concat(out, axis=1)
        return out

def loadModel(net, model_path):
    if net == 'unet':
        model = UNet(13)
    if net == 'pspnet':
        model = PSPNet()
    params_dict = paddle.load(model_path)
    model.set_state_dict(params_dict)
    return model


def Val(net='unet'):
    image_folder = r"work/dataset/val_frames"
    label_folder = r"work/dataset/val_masks"
    model_path = r"work/Class3/output/{}_epoch200.pdparams".format(net)
    output_dir = r"work/Class3/val_result"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model = loadModel(net, model_path)
    model.eval()
    dataset = Basic_ValDataset(image_folder, label_folder, 256)  # size为256
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    result_dict = {}
    val_acc_list = []
    val_iou_list = []

    for index, data in enumerate(dataloader):
        image = data["Image"]
        label = data["Label"]
        imgPath = data["Path"][0]
        image = paddle.transpose(image, [0, 3, 1, 2])
        pred = model(image)
        label_pred = np.argmax(pred.numpy(), 1)
        # 计算acc和iou指标
        label_true = label.numpy()
        acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(label_true, label_pred, n_class=13)
        filename = imgPath.split('/')[-1]
        print('{}, acc:{}, iou:{}, acc_cls{}'.format(filename, acc, mean_iu, acc_cls))
        val_acc_list.append(acc)
        val_iou_list.append(mean_iu)
        result = label_pred[0]
        cv2.imwrite(os.path.join(output_dir, filename), result)

    val_acc, val_iou = np.mean(val_acc_list), np.mean(val_iou_list)
    print('val_acc:{}, val_iou{}'.format(val_acc, val_iou))

def mask2color(mask, labels):
    jsonfile = json.load(open(r"work/dataset/mask2color.json"))
    h, w = mask.shape[:2]
    color = np.zeros([h, w, 3])

    for index in range(len(labels)):
        if index>=8:
            mask_index = index+1 # mask标签需要改变
        else:
            mask_index = index

        if mask_index in mask:
            color[np.where(mask == mask_index)] = np.asarray(jsonfile[labels[index]])
        else:
            continue
    return color

# 将转换好的color图保存在Class2文件夹下的val_color_result文件夹
def save_color(net):
    save_dir = r"work/Class3/{}_color_result".format(net)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    mask_dir = r"work/Class3/val_result"
    mask_names = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    for maskname in mask_names:
        mask = cv2.imread(os.path.join(mask_dir, maskname), -1)
        color = mask2color(mask, labels)
        result = Image.fromarray(np.uint8(color))
        result.save(os.path.join(save_dir, maskname))

newsize = (256, 256)
gt_color1 = Image.open(r"work/dataset/val_colors/000000000GT.png").resize(newsize)
frames1 = Image.open(r"work/dataset/val_frames/000000000.png").resize(newsize)
unet1 = Image.open(r"work/Class3/unet_color_result/000000000.png")
pspnet1 = Image.open(r"work/Class3/pspnet_color_result/000000000.png")

gt_color2 = Image.open(r"work/dataset/val_colors/000000032GT.png").resize(newsize)
frames2 = Image.open(r"work/dataset/val_frames/000000032.png").resize(newsize)
unet2 = Image.open(r"work/Class3/unet_color_result/000000032.png")
pspnet2 = Image.open(r"work/Class3/pspnet_color_result/000000032.png")

gt_color3 = Image.open(r"work/dataset/val_colors/000000041GT.png").resize(newsize)
frames3 = Image.open(r"work/dataset/val_frames/000000041.png").resize(newsize)
unet3 = Image.open(r"work/Class3/unet_color_result/000000041.png")
pspnet3 = Image.open(r"work/Class3/pspnet_color_result/000000041.png")

plt.figure(figsize=(20,24))#设置窗口大小
plt.subplot(3,4,1), plt.title('frames')
plt.imshow(frames1), plt.axis('off')
plt.subplot(3,4,2), plt.title('GT')
plt.imshow(gt_color1), plt.axis('off')
plt.subplot(3,4,3), plt.title('unet')
plt.imshow(unet1), plt.axis('off')
plt.subplot(3,4,4), plt.title('pspnet')
plt.imshow(pspnet1), plt.axis('off')

plt.subplot(3,4,5), plt.title('frames')
plt.imshow(frames2), plt.axis('off')
plt.subplot(3,4,6), plt.title('GT')
plt.imshow(gt_color2), plt.axis('off')
plt.subplot(3,4,7), plt.title('unet')
plt.imshow(unet2), plt.axis('off')
plt.subplot(3,4,8), plt.title('pspnet')
plt.imshow(pspnet2), plt.axis('off')

plt.subplot(3,4,9), plt.title('frames')
plt.imshow(frames3), plt.axis('off')
plt.subplot(3,4,10), plt.title('GT')
plt.imshow(gt_color3), plt.axis('off')
plt.subplot(3,4,11), plt.title('unet')
plt.imshow(unet3), plt.axis('off')
plt.subplot(3,4,12), plt.title('pspnet')
plt.imshow(pspnet3), plt.axis('off')

plt.show()