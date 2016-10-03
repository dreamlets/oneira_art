--add Torch dependencies 
require 'nn'
require 'image'

--add model dependencies
require 'nngraph' 
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

args = parser:parse()

in_folder = args.input
out_folder = args.output
dataset_size = tonumber(args.size)
extension = args.extension

--define dataset and model options
batch_size = 10 
channels = args.channels
dim = args.dimensions

--load model, build dataset and reshape tensors
model = torch.load('model/BR_DCGAN_4000_VAE.t7')
dataset = torch.FloatTensor(dataset_size, channels, dim, dim)
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
    if file ~= '.' and file ~= '..' then
        dataset[count] = image.load(in_folder .. file)
        count = count + 1
    end
end

tm = torch.Timer()

for i = 1, dataset_size, batch_size do 
    xlua.progress(1, dataset_size)
    local size = math.min(i + batch_size - 1, dataset_size) - i
    local input_x = dataset:narrow(1, size, batch_size)
    local samples, __ = table.unpack(model:forward(input_x))
    samples = reshape:forward(samples)
    print('Current memory usage: ' .. collectgarbage("count"))
    print('Saving batch samples...') 
    for idx = 1, batch_size do 
        image.save(out_folder .. 'Sample' .. (idx + i) .. '.' .. extension, samples[idx])
    end
end

print('Completed forward pass in: ' .. tm:time().real) 
print('Samples saved at ' .. out_folder) 
