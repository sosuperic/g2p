-- Convert model from GPU to CPU, save with same name but with _CPU
-- Note: not every model can be converted to CPU. Some modules, e.g. cudnn.LSTM are only on GPU

require 'model_google'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:option('-model_path', '', 'Path to model')
cmd:option('-gpuid', 0, 'ID of gpu to run on')
local opt = cmd:parse(arg)

require 'cunn'
require 'cutorch'
require 'cudnn'
cutorch.setDevice((3 - opt.gpuid) + 1)
cutorch.manualSeed(123)

print('Loading model')
local model = torch.load(opt.model_path)
print('Converting model to double')
model = model:double()
local outpath = opt.model_path:sub(1,#opt.model_path -3) .. '_CPU.t7'
print('Saving model to ' .. outpath)
torch.save(outpath, model)