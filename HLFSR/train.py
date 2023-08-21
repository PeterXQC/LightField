import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
import torchvision.transforms as transforms
import scipy


def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)


    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))


        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


        ''' Validation '''
        step = 1
        if (idx_epoch + 1)%step==0 or idx_epoch > args.epoch-step:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, idx_epoch, save_dir)
                    excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

                    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                    ssim_epoch_test = float(np.array(ssim_iter_test).mean())


                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                    idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                psnr_mean_test = float(np.array(psnr_testset).mean())
                ssim_mean_test = float(np.array(ssim_testset).mean())
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                                  % (psnr_mean_test, ssim_mean_test))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass


def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()
        data = data.to(device)      # low resolution
        my_1,my_2,my_x_dim,my_y_dim = label.shape
        label = label[:,:,my_x_dim//5*2:my_x_dim//5*3,my_y_dim//5*2:my_y_dim//5*3]

        label = label.to(device)    # high resolution
        out, _, _, _, _, _ = net(data, data_info)

        # my_tmp = np.squeeze(label[0,:,:,:].cpu().detach().numpy())
        # plt.imshow(my_tmp)
        # plt.show()

        loss = criterion(out, label, data_info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        pass
        # break

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train




def test(test_loader, device, net, idx_epoch, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()
        my_1, my_2, my_x_dim, my_y_dim = Hr_SAI_y.shape
        Hr_SAI_y_center = Hr_SAI_y[:,:,my_x_dim//5*2:my_x_dim//5*3,my_y_dim//5*2:my_y_dim//5*3]

        # my_tmp = subLFout[0,0,:,:].cpu().detach().numpy()
        # plt.imshow(my_tmp)
        # plt.show()

        subLFin = Lr_SAI_y
        ''' SR the Patches '''
        for i in range(1):
            tmp = subLFin
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out, HFEM_1_out, HFEM_2_out, HFEM_3_out, HFEM_4_out, HFEM_5_out = net(tmp.to(device), data_info)

                subLFout= out

        ''' Calculate the PSNR & SSIM '''
        try:
            psnr, ssim = cal_metrics(args, Hr_SAI_y_center, subLFout)
        except Exception as e:
            print(f"Error during metric calculation: {e}")
            psnr, ssim = 0, 0
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])

        ''' Save mid-result '''

        if save_dir is not None and idx_epoch % 10 == 0:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)

            # Save HFEM outputs
            for i, HFEM_out in enumerate([HFEM_1_out, HFEM_2_out, HFEM_3_out, HFEM_4_out, HFEM_5_out], 1):
                HFEM_out = HFEM_out.cpu().detach()  # Detach from the computation graph and move to CPU memory
                HFEM_out_dict = {f"channel_{j}": HFEM_out[0, j, :, :].numpy() for j in range(HFEM_out.shape[1])}  # Create a dictionary with all channels
                scipy.io.savemat(save_dir_.joinpath(f"{LF_name[0]}_HFEM_{i}_out_{idx_epoch}.mat"), HFEM_out_dict)  # Save the dictionary to a .mat file
            
        print('saved')
        pass

        '''save out'''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            img = torch.squeeze(torch.clamp(subLFout, 0, 1))
            img = transforms.ToPILImage()(img)
            img.save(f"{LF_name[0]}_CenterView_{idx_epoch}.png", format='PNG')
            # imageio.imwrite(path, img)
        pass

    return psnr_iter_test, ssim_iter_test, LF_iter_test


if __name__ == '__main__':
    from option import args

    args.model_name = 'HAT_HLFSR'
    args.angRes = 5
    args.scale_factor = 4
    args.batch_size = 1
    args.patch_size_for_test=1

    main(args)

    print('end')
