import torch
import pickle
import time
import tqdm
import json
from torch import nn, optim
from models import *
a_file = open("datadims.pkl","rb")

dataDims = pickle. load(a_file)

import argparse
import json
def transform_data_2(sample,sample_identity):

    newdata = np.zeros((sample.shape[0], 80, 80, 80))
    for i in range(0, len(sample)):
        for j in range(0, sample.shape[1]):
                if np.isnan(sample[i,j,:]).any() == False and sum(sample[i,j,:]>100)<1:
                   if sample_identity[i,j]=='H':

                        newdata[i, int((sample[i , j, 2])/ 0.25), int((sample[i, j, 1])/ 0.25), int((sample[i , j, 0])/ 0.25)] = 1
                   if sample_identity[i,j] == 'O':
                        newdata[i, int((sample[i, j, 2]) / 0.25), int((sample[i, j, 1]) / 0.25), int((sample[i, j, 0]) / 0.25)] = 2
    return newdata


from pathlib import Path

# --- Resolve path to mainparameters.py whether run from file or notebook ---
if "__file__" in globals():
    # When run as a script
    main_dir = Path(__file__).resolve().parent
else:
    # When run interactively (e.g. Jupyter)
    main_dir = Path.cwd()  # assumes main.py is in current directory

sys.path.insert(0, str(main_dir))

# Import main.py and grab its parser
import mainparameters # model architecture
parser = mainparameters.parser  # reuse the ArgumentParser defined there

# Parse CLI args 
#args = parser.parse_args([])  # or sys.argv[1:] if running from CLI

configs, unknown = parser.parse_known_args()

#samples_train = np.load('data/water.npy', allow_pickle=True)
#samples_identity = np.load('data/water_identity.npy', allow_pickle=True)


#samples_train= transform_data_2(samples_train,samples_identity)
      
#samples_train= np.expand_dims(samples_train, axis=1)
#num_conditioning_variables = samples_train.shape[1] - 1
#assert samples_train.ndim == 5



model = GatedPixelCNN(configs,dataDims)
device = torch.device('cuda:0')
model.eval()
model.to(torch.device("cuda:0"))

optimizer = optim.AdamW(model.parameters(),amsgrad=True) #optimizer = optim.AdamW(model.parameters(),amsgrad=True)
checkpoint = torch.load('model-16.pt', map_location=device)
bc_old=checkpoint['model_state_dict']
bc_new=bc_old.copy()
for items in bc_old.items():
    s1 = (items[0])
    s2 = s1[7:]
   
    bc_new[s2] = bc_new.pop(s1)
model.load_state_dict(bc_new)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if configs.sample_generation_mode == 'serial':
    time_ge = time.time()
    # set dimension of generated sample, can be done according to dataDims values or arbitrary number of voxels
    dataDims['sample x dim']=50#int(dataDims['sample x dim']*3/3)
    dataDims['sample y dim']=50#int(dataDims['sample y dim']*3/3)
    dataDims['sample z dim']=50#int(dataDims['sample z dim']*3/3)
    sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
    sample_y_padded = dataDims['sample y dim'] + 2 * dataDims[
        'conv field'] * configs.boundary_layers  # don't need to pad the bottom
    sample_z_padded = dataDims['sample z dim'] + dataDims['conv field'] * configs.boundary_layers
    sample_conditions = dataDims['num conditioning variables']

    batches = int(np.ceil(configs.n_samples / configs.sample_batch_size))
    # n_samples = sample_batch_size * batches
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample z dim'],
                         dataDims['sample y dim'],
                         dataDims['sample x dim'])  # sample placeholder
    sample =(sample).clone().detach()
    #samples_train=torch.tensor(samples_train,device=device).float()

    print('Generating {} Samples'.format(configs.n_samples))

    for batch in range(batches):  # can't do these all at once so we do it in batches
        print('Batch {} of {} batches'.format(batch + 1, batches))
        sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] + sample_conditions,
                                         sample_z_padded + 1 * dataDims['conv field'] + 2,
                                         sample_y_padded + 2 * dataDims['conv field'] + 1,
                                         sample_x_padded + 2 * dataDims[
                                             'conv field'])  # needs to be explicitly padded by the convolutional field
        sample_batch.fill_(0)  # initialize with minimum value
# can set the boundaries to alternative classes, like padding class (0)
#        sample_batch[:, :, 0:dataDims['conv field'] + 2, :, :] = (0)
#        sample_batch[:, :, :, 0:dataDims['conv field'] + 1, :] = (0)
#        sample_batch[:, :, :, :, 0:dataDims['conv field']] = (0)



        #   if configs.do_conditioning: # assign conditions so the model knows what we want
        #      for i in range(len(configs.generation_conditions)):
        #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']
        print([sample_batch.shape, sample.shape])

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
                                    i - dataDims['conv field']:i + dataDims['conv field'] + 1].cuda())
                       # print([k + 1,j + dataDims['conv field'] * (1 - 0) + 1,i + dataDims['conv field'] + 1])
                        out = torch.reshape(out, (
                            out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-3],
                            out.shape[-2],
                            out.shape[-1]))  # reshape to select channels
                        # print(out.shape)
                        probs = F.softmax(out[:, 1:, 0, -1, -dataDims['conv field'] - 1, dataDims['conv field']],
                                          dim=1).data  # the remove the lowest element (boundary)

                        #  print(sample_batch.shape)
                        sample_batch[:, 0, k, j, i] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / \
                                                      dataDims['classes']  # convert output back to training space
                     #   print([k,j,i])
                        del out, probs

        print('check')
        for k in range(dataDims['channels']):
            sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :,
            :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 2:,
                 (configs.boundary_layers + 1) * dataDims['conv field'] + 1:-(
                         (configs.boundary_layers + 1) * dataDims['conv field']),
                 (configs.boundary_layers + 1) * dataDims['conv field']:-(
                         (configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims[
                     'classes'] - 1  # convert back to input space

    
    np.save('samples/sample_{}'.format(configs.run_num), sample.cpu())


