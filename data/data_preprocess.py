import torch
import torch.nn as nn
from torchvision import transforms


def std_norm(image):  # input tensor image size with CxHxW
    image = image.permute(1, 2, 0).numpy()  # Convert to HWC

    mean = torch.tensor(image).mean(dim=[0, 1])
    std = torch.tensor(image).std(dim=[0, 1])

    # ⚠️ Replace zero std values with 1.0 to avoid division by zero
    std[std == 0] = 1.0

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return trans(image)



def one_zero_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    # data = image.view(channel, height*width)
    data = image.reshape(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1)) / (data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel

    return data.view(channel, height, width)


def pos_neg_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = -1 + 2 * (data - data_min.unsqueeze(1)) / (data_max.unsqueeze(1) - data_min.unsqueeze(1))

    return data.view(channel, height, width)


def construct_sample(img1, img2, img_gt, x1_noisy, window_size=5):
    '''
        function：construct sample
        input: image
                window_size
        output：pad_img1, pad_img2, batch_indices
    '''
    _, height, width = img1.shape
    half_window = int(window_size // 2)  # 2
    pad = nn.ReplicationPad2d(half_window)
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)
    pad_gt = pad(img_gt.unsqueeze(0)).squeeze(0)
    x1_noisy_pad = pad(x1_noisy.unsqueeze(0)).squeeze(0)

    patch_coordinates = torch.zeros((height * width, 4), dtype=torch.long)  # torch.Size([111583, 4])
    t = 0
    for h in range(height):
        for w in range(width):
            patch_coordinates[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_img1, pad_img2, pad_gt, x1_noisy_pad, patch_coordinates


def label_transform(gt):
    #gt   tensor([  0., 255.]) -> # tensor([0., 1.])
    gt_new = torch.zeros_like(gt)
    indices = torch.where(gt == 255)
    gt_new[indices] = 1

    return gt_new


def label_transform012(gt):
    #gt[0. 1. 2.]  -> gt[2. 1. 0.]
    gt_new = torch.zeros_like(gt)
    indices = torch.where(gt == 0)
    gt_new[indices] = 2
    indices = torch.where(gt == 1)
    gt_new[indices] = 1
    indices = torch.where(gt == 2)
    gt_new[indices] = 0

    return gt_new


def label_transform52(gt):
    # gt[0. 1. 2. 3. 4. 5.]  -> gt[0. 1.]'
    gt_new = torch.zeros_like(gt)
    indices = torch.where(gt == 1)
    gt_new[indices] = 1
    indices = torch.where(gt == 2)
    gt_new[indices] = 1
    indices = torch.where(gt == 3)
    gt_new[indices] = 1
    indices = torch.where(gt == 4)
    gt_new[indices] = 1
    indices = torch.where(gt == 5)
    gt_new[indices] = 1

    return gt_new


def select_sample(gt, ntr):
    gt_vector = gt.reshape(-1, 1).squeeze(1)
    label = torch.unique(gt)

    first_time = True

    for each in range(2):
        indices_vector = torch.where(gt_vector == label[each])
        indices = torch.where(gt == label[each])

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        class_num = torch.tensor(len(indices_vector))
        ntr_trl = ntr[0]
        ntr_tru = ntr[1]

        if ntr_trl < 1:
            select_num_trl = int(ntr_trl * class_num)
        else:
            select_num_trl = int(ntr_trl)

        if ntr_tru < 1:
            select_num_tru = int(ntr_tru * class_num)
        else:
            select_num_tru = int(ntr_tru)

        select_num_trl = torch.tensor(select_num_trl)  # River 不变0->20377, 变化1->1939
        select_num_tru = torch.tensor(select_num_tru)

        # disorganize
        torch.manual_seed(1)  #
        rand_indices0 = torch.randperm(class_num)
        rand_indices = indices_vector[rand_indices0]

        # Divide train and test
        trl_ind0 = rand_indices0[0:select_num_trl]  # trl train label 训练集，有标记样本
        tru_ind0 = rand_indices0[select_num_trl:select_num_trl + select_num_tru]  # tru train unlabel 训练集，无标记样本
        te_ind0 = rand_indices0[select_num_trl + select_num_tru:]  # test 测试集
        trl_ind = rand_indices[0:select_num_trl]
        tru_ind = rand_indices[select_num_trl:select_num_trl + select_num_tru]
        te_ind = rand_indices[select_num_trl + select_num_tru:]

        # trl_data：index+Sample center
        select_trl_ind = torch.cat([trl_ind.unsqueeze(1),
                                    indices_row[trl_ind0].unsqueeze(1),
                                    indices_column[trl_ind0].unsqueeze(1)],
                                   dim=1
                                   )  # torch.Size([x, 3])
        # tru_data
        select_tru_ind = torch.cat([tru_ind.unsqueeze(1),
                                    indices_row[tru_ind0].unsqueeze(1),
                                    indices_column[tru_ind0].unsqueeze(1)],
                                   dim=1
                                   )
        # test_data
        select_te_ind = torch.cat([te_ind.unsqueeze(1),
                                   indices_row[te_ind0].unsqueeze(1),
                                   indices_column[te_ind0].unsqueeze(1)],
                                  dim=1
                                  )

        if first_time:
            first_time = False

            trainl_sample_center = select_trl_ind
            trainl_sample_num = select_num_trl.unsqueeze(0)

            trainu_sample_center = select_tru_ind
            trainu_sample_num = select_num_tru.unsqueeze(0)

            test_sample_center = select_te_ind
            test_sample_num = (class_num - select_num_trl - select_num_tru).unsqueeze(0)

        else:
            trainl_sample_center = torch.cat([trainl_sample_center, select_trl_ind], dim=0)
            trainl_sample_num = torch.cat([trainl_sample_num, select_num_trl.unsqueeze(0)])

            trainu_sample_center = torch.cat([trainu_sample_center, select_tru_ind], dim=0)
            trainu_sample_num = torch.cat([trainu_sample_num, select_num_tru.unsqueeze(0)])

            test_sample_center = torch.cat([test_sample_center, select_te_ind], dim=0)
            test_sample_num = torch.cat(
                [test_sample_num, (class_num - select_num_trl - select_num_tru).unsqueeze(0)])

        rand_trl_ind = torch.randperm(trainl_sample_num.sum())
        trainl_sample_center = trainl_sample_center[rand_trl_ind,]
        rand_tru_ind = torch.randperm(trainu_sample_num.sum())
        trainu_sample_center = trainu_sample_center[rand_tru_ind,]
        rand_te_ind = torch.randperm(test_sample_num.sum())
        test_sample_center = test_sample_center[rand_te_ind,]

    data_sample = {'trainl_sample_center': trainl_sample_center, 'trainl_sample_num': trainl_sample_num,
                   'trainu_sample_center': trainu_sample_center, 'trainu_sample_num': trainu_sample_num,
                   'test_sample_center': test_sample_center, 'test_sample_num': test_sample_num,
                   }

    return data_sample
