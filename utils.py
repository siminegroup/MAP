# metrics to determine the performance of our learning algorithm

import numpy as np
import torch.nn.functional as F
import os
from torch import nn, optim, cuda, backends
import torch
from torch.utils import data
import time
from torch.utils.data import Dataset
import pickle
import sys
import tqdm
from tqdm import tqdm as barthing
from models import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



import h5py


class build_dataset(Dataset):
    def __init__(self, configs):
        np.random.seed(configs.dataset_seed)
        self.samples = np.load('training_samples/water.npy', allow_pickle=True)

        self.samples_identity = np.load('training_samples/water_identity.npy', allow_pickle=True)

        #self.samples= self.samples[0:300]
        self.samples = transform_data(self.samples,self.samples_identity)###voxelize data
        self.samples = np.expand_dims(self.samples, axis=1)
        self.num_conditioning_variables = self.samples.shape[1] - 1
        assert self.samples.ndim == 5
        
        ##### Data Augmentation,
        self.samples=np.concatenate((self.samples[:,:,0:40,0:40,0:40],self.samples[:,:,40:,40:,40:],self.samples[:,:,40:,0:40,0:40],self.samples[:,:,0:40,40:,0:40],self.samples[:,:,0:40,0:40,40:],self.samples[:,:,0:40,40:,40:],self.samples[:,:,40:,0:40,40:],self.samples[:,:,40:,40:,0:40]))
        ### not all of these augmentation are needed really
        rot=np.rot90(self.samples.copy(),k=1,axes=(2,3))
        rot2=np.rot90(self.samples.copy(),k=2,axes=(2,3))
        rot3=np.rot90(self.samples.copy(),k=1,axes=(2,4))
        rot4=np.rot90(self.samples.copy(),k=2,axes=(2,4))
        rot5=np.rot90(self.samples.copy(),k=1,axes=(3,4))
        rot6=np.rot90(self.samples.copy(),k=2,axes=(3,4))
        self.samples = np.concatenate((self.samples, rot,rot2,rot3), axis=0) 
        
        np.random.shuffle(self.samples)
        self.dataDims = {
            'classes' : len(np.unique(self.samples)),
            'input x dim' : self.samples.shape[-1],
            'input y dim' : self.samples.shape[-2],
            'input z dim': self.samples.shape[-3],
            'channels' : 1, # hardcode as one so we don't get confused with conditioning variables
            'dataset length' : len(self.samples),
            'sample x dim' : self.samples.shape[-1] * configs.sample_outpaint_ratio,
            'sample y dim' : self.samples.shape[-2] * configs.sample_outpaint_ratio,
            'sample z dim': self.samples.shape[-3] * configs.sample_outpaint_ratio,
            'num conditioning variables' : self.num_conditioning_variables,
            'conv field' : configs.conv_layers + configs.conv_size // 2,
           # 'conditional mean' : self.conditional_mean,
           # 'conditional std' : self.conditional_std
        }
        a_file = open("datadims.pkl", "wb")
        pickle.dump(self.dataDims, a_file)
        a_file.close()
	
        # normalize pixel inputs
        self.samples[:,0,:,:,:] = np.array((self.samples[:,0] + 1)/(self.dataDims['classes'])) # normalize inputs on 0--1

        torch.cuda.empty_cache()
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dir_name(model, training_data, filters, layers, dilation, filter_size, noise, den_var, dataset_size):
    dir_name = "model=%d_dataset=%d_dataset_size=%d_filters=%d_layers=%d_dilation=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, dataset_size, filters, layers, dilation, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved

    return dir_name


def get_model(configs, dataDims):
    if configs.model == 'gated1':
        model = GatedPixelCNN(configs, dataDims) # gated, without blind spot
    else:
        sys.exit()

    return model


    #def init_weights(m):
    #    if (type(m) == nn.Conv2d) or (type(m) == MaskedConv2d):
    #        #torch.nn.init.xavier_uniform_(m.weight)
    #        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity = 'relu')

    #model.apply(init_weights) # apply xavier weights to 1x1 and 3x3 convolutions


def get_dataloaders(configs):
    dataset = build_dataset(configs)  # get data
    dataDims = dataset.dataDims
    print(['dataset',len(dataset)])
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    tr = data.DataLoader(train_dataset, batch_size=configs.training_batch_size, shuffle=True, num_workers= 0, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers= 0, pin_memory=True)
    print([type(tr),len(te)])
    return tr, te, dataDims


def initialize_training(configs):
    dist_url = "env://" # default

               
            
    rank=0
    world_size=1
    dist.init_process_group(backend="nccl", init_method=configs.init_method, rank=rank, world_size=world_size)
    
    tr, te, dataDims = get_dataloaders(configs)
    model = get_model(configs, dataDims)
    model= model.to(rank)
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    dataDims['conv field'] = configs.conv_layers + configs.conv_size // 2

    optimizer =  optim.SGD(ddp_model.parameters(),lr=1e-2, momentum=0.9, nesterov=True)#optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#optim.AdamW(ddp_model.parameters(),lr=0.01, amsgrad=True)

    return ddp_model, optimizer, dataDims


def compute_loss(output, target):
    target = target[:,:1]
    lossi = []
    lossi.append(F.cross_entropy(output, target.squeeze(1).long()))
    return torch.sum(torch.stack(lossi))


def get_training_batch_size(configs, model):
    finished = 0
    training_batch_0 = 1 * configs.training_batch_size
    #  test various batch sizes to see what we can store in memory
    dataset = build_dataset(configs)
    dataDims = dataset.dataDims
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    optimizer = optim.SGD(model.parameters(),lr=1e-2, momentum=0.9, nesterov=True)#optim.AdamW(model.parameters(),amsgrad=True) #

    while (configs.training_batch_size > 1) & (finished == 0):
        try:
            test_dataloader = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers=0, pin_memory=True)
            model_epoch(configs, dataDims = dataDims, trainData = test_dataloader, model = model, optimizer = optimizer, update_gradients = True, iteration_override = 2)
            finished = 1
        except RuntimeError: # if we get an OOM, try again with smaller batch
            configs.training_batch_size = int(np.ceil(configs.training_batch_size * 0.8)) - 1
            print('Training batch sized reduced to {}'.format(configs.training_batch_size))

    return max(int(configs.training_batch_size * 0.25),1), int(configs.training_batch_size != training_batch_0)


def model_epoch(configs, dataDims = None, trainData = None, model=None, optimizer=None, update_gradients = True, iteration_override = 0):
    if configs.CUDA:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()

    err = []

    if update_gradients:
        model.train(True)
    else:
        model.eval()
    print(['traindata',len(trainData)])
    for i, input in enumerate(trainData):

        if configs.CUDA:
            input = input.cuda(non_blocking=True)

        target = input * dataDims['classes']

        output = model(input.float()) # reshape output from flat filters to channels * filters per channel
        loss = compute_loss(output, target)

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override != 0:
            if i > iteration_override:
                break

    print(i)
    if configs.CUDA:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err, time_tr

def model_epoch_new(configs, dataDims = None, trainData = None, model=None, optimizer=None, update_gradients = True, iteration_override = 0):
    # if configs.CUDA:
    #     cuda.synchronize()  # synchronize for timing purposes
    # time_tr = time.time()
   # model=model.to(rank)
    
   # ddp_model = DDP(model, device_ids=[rank])
    if configs.CUDA:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()


    err = []
    rank=0

    if update_gradients:
        model.train(True)
    else:
        model.eval()
    print(['traindata',len(trainData)])
    for i, input in enumerate(trainData):

        # if configs.CUDA:
        #     input = input.cuda(non_blocking=True)

        target = (input * dataDims['classes']).to(rank)

        output = model(input.float().to(rank)) # reshape output from flat filters to channels * filters per channel
        loss = compute_loss(output, target)

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override != 0:
            if i > iteration_override:
                break

    print(i)
    if configs.CUDA:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err, time_tr





def auto_convergence(configs, epoch, tr_err_hist, te_err_hist):
    # set convergence criteria
    # if the test error has increased on average for the last x epochs
    # or if the training error has decreased by less than 1% for the last x epochs
    train_margin = .000001  # relative change over past x runs
    # or if the training error is diverging from the test error by more than 20%
    test_margin = 10 # max divergence between training and test losses
    configs.convergence_moving_average_window #the time over which we will average loss in order to determine convergence
    converged = 0
    if epoch > configs.convergence_moving_average_window:
     
        window = configs.convergence_moving_average_window
        tr_mean, te_mean = [torch.mean(torch.stack(tr_err_hist[-configs.convergence_moving_average_window:])), torch.mean(torch.stack(te_err_hist[-configs.convergence_moving_average_window:]))]
        print([tr_mean,tr_err_hist[-window],configs.convergence_margin,te_mean])
        if (torch.abs((tr_mean - tr_err_hist[-window]) / tr_mean) < configs.convergence_margin) \
                or ((torch.abs(te_mean - tr_mean) / tr_mean) < configs.convergence_margin) \
                or (epoch == configs.max_epochs)\
                or (te_mean > te_err_hist[-window]):
            converged = 1
            print('Learning converged at epoch {}'.format(epoch - window))  # print a nice message  # consider also using an accuracy metric

    return converged


def generate_samples_gated(configs, dataDims, model):
    if configs.sample_generation_mode == 'serial':
        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + 2 * dataDims[
            'conv field'] * configs.boundary_layers  # don't need to pad the bottom
        sample_z_padded = dataDims['sample z dim'] + dataDims['conv field'] * configs.boundary_layers
        sample_conditions = dataDims['num conditioning variables']

        batches = int(np.ceil(configs.n_samples / configs.sample_batch_size))
        # n_samples = sample_batch_size * batches
        sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample z dim'],
                             dataDims['sample y dim'],
                             dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] + sample_conditions,
                                             sample_z_padded + 1 * dataDims['conv field'] + 2,
                                             sample_y_padded + 2 * dataDims['conv field'] + 1,
                                             sample_x_padded + 2 * dataDims[
                                                 'conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            #   if configs.do_conditioning: # assign conditions so the model knows what we want
            #      for i in range(len(configs.generation_conditions)):
            #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']
            print([sample_batch.shape, sample.shape])
            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            # generator.train(False)
            model.eval()

            with torch.no_grad():  # we will not be updating weights
                for k in tqdm.tqdm(
                        range(dataDims['conv field'] + 2,
                              sample_z_padded + dataDims['conv field'] + 2)):  # for each pixel
                    for j in range(dataDims['conv field'] + 1, sample_y_padded + dataDims['conv field'] + 1):
                        for i in range(dataDims['conv field'], sample_x_padded + dataDims['conv field']):
                            # out = generator(sample_batch.float())
                            out = model(sample_batch[:, :, k - dataDims['conv field'] - 2:k + 1,
                                        j - dataDims['conv field'] - 1:j + dataDims['conv field'] * (1 - 0) + 1,
                                        i - dataDims['conv field']:i + dataDims['conv field'] + 1].float())
                            out = torch.reshape(out, (
                                out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-3],
                                out.shape[-2],
                                out.shape[-1]))  # reshape to select channels
                            # print(out.shape)
                            probs = F.softmax(out[:, 1:, 0, -1, -dataDims['conv field'] - 1, dataDims['conv field']],
                                              dim=1).data  # the remove the lowest element (boundary)
                            #  print(probs.shape)
                            #  print(sample_batch.shape)
                            sample_batch[:, 0, k, j, i] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / \
                                                          dataDims['classes']  # convert output back to training space

                            del out, probs

            for k in range(dataDims['channels']):
                sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :,
                :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 2:,
                     (configs.boundary_layers + 1) * dataDims['conv field'] + 1:-(
                                 (configs.boundary_layers + 1) * dataDims['conv field']),
                     (configs.boundary_layers + 1) * dataDims['conv field']:-(
                                 (configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims[
                         'classes'] - 1  # convert back to input space

        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time() - time_ge


    return sample, time_ge


def generation(configs, dataDims, model):
    #err_te, time_te = test_net(model, te)  # clean run net

    sample, time_ge = generate_samples_gated(configs, dataDims, model)  # generate samples

    np.save('samples/run_{}'.format(configs.run_num), sample)

    if len(sample) != 0:
        print('Generated samples')
        return sample, time_ge

    else:
        print('Sample Generation Failed!')
        return 0, 0, 0, 0



def save_ckpt(epoch, net, optimizer, dir_name):
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/' + dir_name[:])

def load_all_pickles(path):
    outputs = []
    print('loading all .pkl files from',path)
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]
    for f in files:
        if f[-4:] in ('.pkl'):
            name = f[:-4]+'_'+f[-3:]
            print('loading', f, 'as', name)
            with open(path + '/' + f, 'rb') as f:
                outputs.append(pickle.load(f))

    return outputs

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def find_dir():
    found = 0
    ii = 0
    while found == 0:
        ii += 1
        if not os.path.exists('logfiles/run_%d' % ii):
            found = 1

    return ii

def rolling_mean(input, run):
    output = np.zeros(len(input))
    for i in range(len(output)):
        if i < run:
            output[i] = np.average(input[0:i])
        else:
            output[i] = np.average(input[i - run:i])

    return output


#def transform_data(sample):

#    newdata = np.zeros((sample.shape[0], 90, 90, 90))
#    for i in range(0, sample.shape[0]):
#        for j in range(0, sample.shape[1]):
                #print([i,j,int((sample[i + 1, j, 2])/ 0.08), int((sample[i + 1, j, 1])/ 0.08), int((sample[i + 1, j, 0])/ 0.08)])

#                newdata[i, int((sample[i , j, 2])/ 0.2), int((sample[i, j, 1])/ 0.2), int((sample[i , j, 0])/ 0.2)] = 1
#    return newdata

def transform_data(sample,sample_identity):

    newdata = np.zeros((sample.shape[0], 80, 80, 80))
    for i in range(0, len(sample)):
        for j in range(0, sample.shape[1]):
                if np.isnan(sample[i,j,:]).any() == False and sum(sample[i,j,:]>100)<1:
                   if sample_identity[i,j]=='H':

                        newdata[i, int((sample[i , j, 2])/ 0.25), int((sample[i, j, 1])/ 0.25), int((sample[i , j, 0])/ 0.25)] = 1
                   if sample_identity[i,j] == 'O':
                        newdata[i, int((sample[i, j, 2]) / 0.25), int((sample[i, j, 1]) / 0.25), int((sample[i, j, 0]) / 0.25)] = 2
    return newdata




def superscale_image(image, f = 1):
    f = 2
    hi, wi = image.shape
    ny = hi // 2
    nx = wi // 2
    tmp = np.reshape(image, (ny, f, nx, f))
    tmp = np.repeat(tmp, f, axis=1)
    tmp = np.repeat(tmp, f, axis=3)
    tmp = np.reshape(tmp, (hi * f, wi * f))

    return tmp

def log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis):

    if configs.comet:
        for i in range(len(sample)):
            experiment.log_image(np.rot90(sample[i, 0]), name='epoch_{}_sample_{}'.format(epoch, i), image_scale=4, image_colormap='hot')
        experiment.log_metrics(agreements, epoch=epoch)

def log_input_stats(configs, experiment, input_analysis):
    if configs.comet:
        for i in range(len(input_analysis['training samples'])):
            experiment.log_image(np.rot90(input_analysis['training samples'][i]), name = 'training example {}'.format(i), image_scale=4, image_colormap='hot')


def standardize(data):
    return (data - np.mean(data)) / np.sqrt(np.var(data))

