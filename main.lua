--add Torch dependencies 
require 'nn'
require 'cunn'
require 'cutorch'
require 'image'

--add model dependencies
require 'graphnn' 
require 'lib/Sampler'
require 'lib/GaussianCriterion'
require 'lib/KLDCriterion'
require 'lib/VAE'

--add utilities
require 'xlua'
local argparse = require 'argparse'
require 'lfs'

--parse cmd options 
local parser = argparse('oneira main', 'execute forward pass of oneira model on some input image sequence')
parser:option('-i --input', 'destination of input image sequence')
parser:option('-o --output', 'destination of output images')
parser:option('-s --size', 'size of image dataset', '100')
parser:option('-e --extension', 'extension of desired output images (jpg or png)', 'jpg')
parser:option('-c --channels', 'number of channels for input images', '3') 
parser:option('-d --dimensions', 'dimension of images (must be square)', '128')

args = parser.parse()

in_folder = args.input
out_folder = args.output
dataset_size = args.size
extension = args.extension

--define dataset and model options
batch_size = 10 
channels = args.channels
dim = args.dimensions

--load model, build dataset and reshape tensors
model = torch.load('model/BR_DCGAN_4000_VAE.t7')
dataset = torch.Tensor(dataset_size, channels, dim, dim)
reshape = nn.Reshape(channels, dim, dim)

--ensure everything expects torch.Float tensors
model:float() 
dataset:float() 
reshape:float()

count = 1
print('adding photos to tensor...')

for file in lfs.dir(in_folder) do
    xlua.progress(count, dataset_size)
    if count >= dataset_size then 
        break 
    end 
    isImage, sample = pcall(image.load, file)
    if isImage then 
        dataset[count] = sample
    else 
        print('bad file. skipping...')
    end
    count = count + 1 
end
