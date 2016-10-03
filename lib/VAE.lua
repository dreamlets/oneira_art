require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(feature_size, h1_size, h2_size, latent_variable_size)

    local encoder = nn.Sequential()
    encoder:add(nn.View(-1, feature_size))
    encoder:add(nn.Linear(feature_size, h1_size))
    encoder:add(nn.BatchNormalization(h1_size))
    encoder:add(nn.ReLU(true))
    encoder:add(nn.Linear(h1_size, h2_size))
    encoder:add(nn.BatchNormalization(h2_size))
    encoder:add(nn.ReLU(true))
    
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(h2_size, latent_variable_size))
    mean_logvar:add(nn.Linear(h2_size, latent_variable_size))

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(feature_size, h1_size, h2_size, latent_variable_size)
    -- The Decoder
    decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, h2_size))
    decoder:add(nn.BatchNormalization(h2_size))
    decoder:add(nn.ReLU(true))
    decoder:add(nn.Linear(h2_size, h1_size))
    decoder:add(nn.BatchNormalization(h1_size))
    decoder:add(nn.ReLU(true))

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(h1_size, feature_size))
    mean_logvar:add(nn.Linear(h1_size, feature_size))
    decoder:add(mean_logvar)
    
    tanh = nn.ParallelTable()
    tanh:add(nn.Tanh())
    tanh:add(nn.Tanh())
    decoder:add(tanh)

    return decoder
end

function VAE.get_discriminator(latent_variable_size, ndf)
    netD = nn.Sequential()
    -- input is (nc) x 64 x 64
    netD:add(nn.SpatialConvolution(channels, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))

    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 16, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 16)):add(nn.LeakyReLU(0.2, true))

    -- state size: (ndf*8) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 16, 1, 4, 4))
    netD:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
    -- state size: 1

    netD:apply(weights_init)
    
    return netD
end

return VAE
