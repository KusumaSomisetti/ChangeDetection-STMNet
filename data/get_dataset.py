from scipy.io import loadmat


def get_Farmland_dataset():
    data_set_before = loadmat('/kaggle/input/farmland/farm06.mat')['imgh']
    data_set_after = loadmat('/kaggle/input/farmland/farm07.mat')['imghl']
    ground_truth = loadmat('/kaggle/input/farmland/label (1).mat')['label']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_Hermiston_dataset():
    data_set_before = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['T1']
    data_set_after = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['T2']
    ground_truth = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['Binary']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_Bayarea_dataset(cfg):
    mat = loadmat(cfg['data_root'] + 'bayArea_dataset.mat')

    img1 = mat['T1'].astype('float32')           # Time 1 image
    img2 = mat['T2'].astype('float32')           # Time 2 image
    gt = mat['groundTruth'].astype('float32')    # Ground truth

    return img1, img2, gt

def get_dataset(current_dataset,cfg):
    if current_dataset == 'Farmland':
        return get_Farmland_dataset()  # Farmland(450, 140, 155), gt[0. 1.]
    elif current_dataset == 'Bayarea':
        return get_Bayarea_dataset(cfg)  # Bayarea(600, 500, 224), gt[0. 1. 2.]
    elif current_dataset == 'Hermiston':
        return get_Hermiston_dataset()  # Hermiston(307, 241, 154), gt[0. 1.]
