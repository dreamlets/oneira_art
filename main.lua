require 'image'
require 'xlua'
require 'nn'
require 'dpnn'
require 'optim'
require 'lfs'

local VAE = require 'VAE'
local discriminator = require 'discriminator'

local argparse = require 'argparse'
local parser = argparse('oneira art', 'a fine-art generator')
parser:option('-i --input', 'input directory for image dataset')
parser:option('-o --output', 'output directory for generated images')
parser:option('-s --size', 'number of samples generated')
parser:option('-m --model', 'location of model') 

args = parser:parse()

input = args.input
output_folder = args.output
batch_size = args.size
model_path = args.model

torch.setnumthreads(4)

function getNumber(num)
  length = #tostring(num)
  filename = ""
  for i=1, (6 - length) do
    filename = filename .. 0
  end
  filename = filename .. num
  return filename
end

z_dim = 100

model = torch.load(model_path)

--noise to pass through decoder to generate random samples from Z
noise_x = torch.Tensor(batch_size, z_dim, 1, 1)
noise_x:normal(0, 0.01)

epoch_tm = torch.Timer()
tm = torch.Timer()
data_tm = torch.Timer()

noise_x:normal(0, 0.01)
generations = decoder:forward(noise_x)
for i = 1, batch_size do
    image.save(output_folder .. getNumber(i) .. '.png', generations[i])
end

