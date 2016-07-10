require 'nnx'
require 'optim'

require 'utils.lua_utils'
require 'utils.lua_stats_utils'
require 'model'

require 'pl'
require 'socket'
require 'xlua'
require 'csvigo'

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('$ th g2p.lua -use_google_model -lr 0.001 -lr_decay 1e-9 -mom 0.9 -damp 0.9 -nesterov true')
cmd:text('$ th g2p.lua -use_google_model -load_model -load_model_dir 2016_6_23___12_14_32 -load_model_fn e4_2.7609.t7')
cmd:text('Options:')
-- Data
cmd:option('-train_on_valid', false, 'Train on valid instead of training. Used to debug because it is faster')
-- Model 
cmd:option('-batchsize', 8, 'number of examples in minibatch')
cmd:option('-epochs', 100, 'max number of epochs to train for')
-- Load model
cmd:option('-load_model', false, 'start training from existing model')
cmd:option('-load_model_dir', '', 'directory to load model and params from')
cmd:option('-load_model_fn', '', 'fn of model to load')
-- Optimization
cmd:option('-use_rmsprop', false, 'Use RMSprop to optimize')
cmd:option('-use_adam', false, 'Use Adam to optimize')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-lr_decay', 0, 'learning rate decay')
cmd:option('-weight_decay', 0, 'weight decay')
cmd:option('-mom', 0, 'momentum')
cmd:option('-damp', 0, 'dampening')
cmd:option('-nesterov', false, 'Nesterov momentum')
-- Bookkeeping
cmd:option('-models_dir', 'models', 'directory to save models to')
cmd:option('-gpuid', -1, 'ID of gpu to run on')
cmd:option('-save_model_every_epoch', 1, 'how often to save model')
cmd:option('-notes', '', 'String of notes, e.g. using batch norm. To keep track of iterative testing / small modifications')
local opt = cmd:parse(arg)

if opt.gpuid >= 0 then
	require 'cunn'
	require 'cutorch'
	require 'cudnn'
	cutorch.setDevice((3 - opt.gpuid) + 1)
	cutorch.manualSeed(123)
end

---------------------------------------------------------------------------
-- Get data
---------------------------------------------------------------------------
local dataset, grapheme_to_idx, phoneme_to_idx = unpack(require 'cmudict_data')
local num_graphemes = size_of_table(grapheme_to_idx)
local num_phonemes = size_of_table(phoneme_to_idx)

-- print(dataset.valid[35])
-- print(dataset.valid[35].x)
-- print(dataset.valid[35].y)

local get_batcher = require 'batcher'

local train_batcher = nil
if opt.train_on_valid then
	train_batcher = get_batcher(dataset.valid, opt.batchsize, num_graphemes)
else
	train_batcher = get_batcher(dataset.train, opt.batchsize, num_graphemes)
end
local val_batcher = get_batcher(dataset.valid, opt.batchsize, num_graphemes)

---------------------------------------------------------------------------
-- Get model and other setup
---------------------------------------------------------------------------

local ctc_criterion = nn.CTCCriterion(false)
local sgd_params = nil
local model = nil
local cur_epoch = 1
local save_path = nil
local load_params = {}
if opt.load_model then -- Load opt, sgd_params, and model, epoch
	load_params = {
		load_model=true, models_dir=opt.models_dir,
		load_model_dir=opt.load_model_dir, load_model_fn=opt.load_model_fn}
	
	opt = torch.load(path.join(load_params.models_dir, load_params.load_model_dir, 'cmd.t7'))
	sgd_params = torch.load(path.join(load_params.models_dir, load_params.load_model_dir, 'sgd_params.t7'))

	-- opt.lr = 0.001
	-- sgd_params.learningRate = 0.001
	-- opt.epochs = 150

	model = torch.load(path.join(load_params.models_dir, load_params.load_model_dir, load_params.load_model_fn))
	cur_epoch = tonumber(string.match(load_params.load_model_fn, 'e(%d+)_.+.t7')) + 1
	save_path = path.join(load_params.models_dir, load_params.load_model_dir)

	print('Loaded model from: ' .. path.join(load_params.models_dir, load_params.load_model_dir))
	print('Current epoch set to: ' .. cur_epoch)
else
	-- Set optimization parameters
	sgd_params = {
	    learningRate = opt.lr,
	    learningRateDecay = opt.lr_decay,
	    weightDecay = opt.weight_decay,
	    momentum = opt.mom,
	    dampening = opt.damp,
	    nesterov = opt.nesterov
	}

	-- Set up network
	-- Check which model to load
	local g2p_model = require 'model_google'
	if opt.gpuid >= 0 then
		model = g2p_model(true, num_graphemes, num_phonemes, opt.batchsize)
	else
		model = g2p_model(false, num_graphemes, num_phonemes, opt.batchsize)
	end

	-- Create directory (if necessary) to save models to using current time 
	local cur_dt = os.date('*t', socket.gettime())
	local save_dirname = string.format('%d_%d_%d___%d_%d_%d',
		cur_dt.year, cur_dt.month, cur_dt.day,
		cur_dt.hour, cur_dt.min, cur_dt.sec)
	save_path = path.join(opt.models_dir, save_dirname)
	make_dir_if_not_exists(save_path)

	-- Save command line options
	print(opt)
	local fp = path.join(save_path, 'cmd')
	torch.save(fp .. '.t7', opt)
	csvigo.save{path=fp .. '.csv', data=convert_table_for_csvigo(opt)}

	-- Save optimization parameters
	local fp = path.join(save_path, 'sgd_params')
	torch.save(fp .. '.t7', sgd_params)
	csvigo.save{path=fp .. '.csv',  data=convert_table_for_csvigo(sgd_params)}
end

-- Convert to cuda if necessary
if opt.gpuid >= 0 then
	model:cuda()
	ctc_criterion = ctc_criterion:cuda()
end

model:training()
local params, grad_params = model:getParameters()

---------------------------------------------------------------------------
-- Feval for training
---------------------------------------------------------------------------
local function feval(params_new)
	grad_params:zero()
	local inputs, targets, sizes = unpack(train_batcher:next_batch())
	
	if opt.gpuid >= 0 then
		inputs = inputs:cuda()
		sizes = sizes:cuda()
	end

	local predictions = model:forward({ inputs, sizes })
	local cur_batchsize	= #targets											-- Last batch in epoch may be smaller than opt.batchsize
	predictions = predictions:view(-1, cur_batchsize, num_phonemes + 1)		-- CTCCriterion expects seq_length x batch x output_dim
	local loss = ctc_criterion:forward(predictions, targets, sizes)
	
	-- Loss used to blow up for some minibatches when using GPU (this should be fixed, see commits)
	if math.abs(loss) <= 1000 then
		local grad_output = ctc_criterion:backward(predictions, targets)
		if opt.gpuid < 0 then
			grad_output = grad_output:double()
		end
		grad_output = grad_output:view(-1, num_phonemes+1)					-- Model's last module is: (seq_length x batch) x output_dim
		model:backward(inputs, grad_output)
		grad_params:div(inputs:size(1)) 		-- Divide by batchsize
	else
		loss = 0	-- TODO: Not really fair, should be average of previous or somethinga
	end

	-- Gradient clipping
    -- grad_params:clamp(-5, 5)
    -- grad_params:clamp(-opt.grad_clip, opt.grad_clip)

	return loss, grad_params
end

---------------------------------------------------------------------------
-- Basically feval, but just the forward pass to get the loss for validation
---------------------------------------------------------------------------
local function val_eval(params_new)
	grad_params:zero()
	local inputs, targets, sizes = unpack(val_batcher:next_batch())

	if opt.gpuid >= 0 then
		inputs = inputs:cuda()
		sizes = sizes:cuda()
	end

	local predictions = model:forward({ inputs, sizes })
	local cur_batchsize	= #targets											-- Last batch in epoch may be smaller than opt.batchsize
	predictions = predictions:view(-1, cur_batchsize, num_phonemes + 1)		-- CTCCriterion expects seq_length x batch x output_dim
	local loss = ctc_criterion:forward(predictions, targets, sizes)

	-- Loss used to blow up for some minibatches when using GPU (this should be fixed, see commits)
	if math.abs(loss) > 1000 then
		loss = 0
	end
	return loss, grad_params
end


---------------------------------------------------------------------------
-- Training
---------------------------------------------------------------------------
-- Store losses so we can save them to csv
local train_losses = {}
local val_losses = {}
local runtime_per_epoch = {}

if load_params.load_model then
	train_losses = csvigo.load{path=path.join(save_path, 'train_losses.csv'), mode='raw', header=false}
	train_losses = subrange(train_losses, 1, cur_epoch - 1)
	val_losses = csvigo.load{path=path.join(save_path, 'val_losses.csv'), mode='raw', header=false}
	val_losses = subrange(val_losses, 1, cur_epoch - 1)
	runtime_per_epoch = csvigo.load{path=path.join(save_path, 'runtime_per_epoch.csv'), mode='raw', header=false}
	runtime_per_epoch = subrange(runtime_per_epoch, 1, cur_epoch - 1)
end

local start_time = os.time()
local num_train_batches = train_batcher:num_batches()
local num_val_batches = val_batcher:num_batches()

for i=cur_epoch,opt.epochs do
	local epoch_start_time = os.time()
	print(string.format('Epoch: %d', i))

	-- Iterate over training set, add training loss to table
	local avg_train_loss = 0
	model:training()
	for j=1,num_train_batches do
		local _, fs
		if opt.use_adam then
			_, fs = optim.adam(feval, params, sgd_params, {})
		elseif opt.use_rmsprop then
			_, fs = optim.rmsprop(feval, params, sgd_params, {})
		else
			_, fs = optim.sgd(feval, params, sgd_params)
		end

		-- Update running average loss
		local batch_loss = fs[1]
		avg_train_loss = avg_train_loss * (num_train_batches-1) / num_train_batches + batch_loss / num_train_batches
		xlua.progress(j, num_train_batches)
	end
	table.insert(train_losses, {i, avg_train_loss})
	print(string.format('Epoch: %d, Average train loss: %f', i, avg_train_loss))

	-- Iterate over validation set
	local avg_val_loss = 0
	model:evaluate()
	for j=1,num_val_batches do
		local _, fs
		if opt.use_adam then
			_, fs = optim.adam(val_eval, params, sgd_params, {})
		elseif opt.use_rmsprop then
			_, fs = optim.rmsprop(val_eval, params, sgd_params, {})
		else
			_, fs = optim.sgd(val_eval, params, sgd_params)
		end

		local batch_loss = fs[1]
		avg_val_loss = avg_val_loss * (num_val_batches-1) / num_val_batches + batch_loss / num_val_batches
		xlua.progress(j, num_val_batches)
	end
	table.insert(val_losses, {i, avg_val_loss})
	print(string.format('Epoch: %d, Average val loss: %f', i, avg_val_loss))

	-- Calculate epoch run time and add to table
	local epoch_run_time = (os.time() - epoch_start_time) / 60		-- In minutes
	table.insert(runtime_per_epoch, {i, epoch_run_time})

	-- Save model
	-- Save in non-cuda so we can easily load in CPU if we need to
	if (i % opt.save_model_every_epoch == 0) then
		local fn = string.format('e%d_%.4f.t7', i, avg_train_loss)
		local fp = path.join(save_path, fn)
		torch.save(fp, model)
		print('Saving checkpoint model to ' .. fp)
	end

	-- Write losses and runtime to csv
	local train_losses_fp = path.join(save_path, 'train_losses.csv')
	local val_losses_fp = path.join(save_path, 'val_losses.csv')
	local runtime_per_epoch_fp = path.join(save_path, 'runtime_per_epoch.csv')
	csvigo.save{path=train_losses_fp, data=train_losses, verbose=false}
	csvigo.save{path=val_losses_fp, data=val_losses, verbose=false}
	csvigo.save{path=runtime_per_epoch_fp, data=runtime_per_epoch, verbose=false}

	-- if i % 15 == 0 then
	-- 	sgd_params.learningRate = sgd_params.learningRate / 2
	-- end
end


-- Write some runtime stats to file
local end_time = os.time()
local run_stats = {}
run_stats['runtime_in_min'] = (end_time - start_time) / 60
local stats_fp = path.join(save_path, 'run_stats.csv')
csvigo.save{path=stats_fp, data=convert_table_for_csvigo(run_stats)}


