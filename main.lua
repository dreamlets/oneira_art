require 'image'
require 'xlua'
require 'nn'
require 'dpnn'
require 'optim'
require 'lfs'

local argparse = require 'argparse'
local parser = argparse('oneira art', 'a fine-art generator')
parser:option('-o --output', 'output directory for generated images')
parser:option('-s --size', 'number of samples generated')
parser:option('-m --model', 'location of model') 

args = parser:parse()

input = args.input
output_folder = args.output
batch_size = tonumber(args.size)
model_path = args.model

torch.setnumthreads(4)

--pad zeros to the end of generated images, up to four
function getNumber(num)
  length = #tostring(num)
  filename = ""
  for i=1, (4 - length) do
    filename = filename .. 0
  end
  filename = filename .. num
  return filename
end

z_dim = 100

--the model contains a variational autoencoder, with an encoder, variational sampler, and decoder. 
--the decoder is all we need for the generation 
model = torch.load(model_path).modules[3]

--noise to pass through decoder to generate random samples from Z
noise_x = torch.Tensor(batch_size, z_dim, 1, 1):float()
noise_x:normal(0, 0.01)

generations = model:forward(noise_x)

for i = 1, batch_size do
    image.save(output_folder .. getNumber(i) .. '.png', generations[i])
end

