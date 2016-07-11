-- Load a network and evaluate it on the test set

require 'warp_ctc'

require 'model_google'

require 'utils.lua_utils'
require 'utils.g2p_utils'

-------------------------------------------------------------------------------------------
-- Setup
------------------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:option('-gpuid', -1, 'ID of gpu to run on')
cmd:option('-test_on_valid', false, 'Test on valid or test set')
cmd:option('-batchsize', 8, 'Minibatch size')
local opt = cmd:parse(arg)

if opt.gpuid >= 0 then
	require 'cunn'
	require 'cutorch'
	require 'cudnn'
	cutorch.setDevice((3 - opt.gpuid) + 1)
	cutorch.manualSeed(123)
end

print('Loading model')
-- local model = torch.load('models/2016_6_30___13_31_35/e33_0.5850.t7')
-- local model = torch.load('models/test_reload_change_lr_3/e70_0.4748.t7')

-- With fully connected between LSTMs
-- local model = torch.load('models/2016_7_6___13_38_9_RELOAD_2/e96_0.6218.t7')
local model = torch.load('models/e114_0.6188.t7') -- With lr_decay


model:evaluate()

print('Getting data')
local dataset, grapheme_to_idx, phoneme_to_idx = unpack(require 'cmudict_data')
local num_phonemes = size_of_table(phoneme_to_idx)
local num_graphemes = size_of_table(grapheme_to_idx)
local idx_to_grapheme = invert_table(grapheme_to_idx)

-- Add blank symbol for CTC (0 is reserved for blank, see warp-ctc tutorial)
phoneme_to_idx['-'] = 0
local idx_to_phoneme = invert_table(phoneme_to_idx)

local get_batcher = require 'batcher'
local batcher = nil
if opt.test_on_valid then
	batcher = get_batcher(dataset.valid, opt.batchsize, num_graphemes)
else
	batcher = get_batcher(dataset.test, opt.batchsize, num_graphemes)
end

------------------------------------------------------------------------------------------
-- Testing
------------------------------------------------------------------------------------------
-- Both phonemes and predictions are tables of table, each entry in nested table is phoneme
-- Return number correct and number incorrect
local function check_wer(phonemes, preds)
	local num_correct, num_wrong = 0, 0
	local matches = {}

	for i=1,#phonemes do
		local wrong = false
		if #phonemes[i] ~= #preds[i] then	-- Short circuit. Must have same number of phonemes
			wrong = true
		else
			for j=1,#phonemes[i] do 		-- Go through phoneme by phoneme
				if phonemes[i][j] ~= preds[i][j] then
					wrong = true
				end
			end
		end

		if wrong then
			num_wrong = num_wrong + 1
			table.insert(matches, 0)
		else
			num_correct = num_correct + 1
			table.insert(matches, 1)
		end
	end
	return {num_correct, num_wrong, matches}
end

------------------------------------------------------------------------------------------
-- Testing
------------------------------------------------------------------------------------------
local num_correct, num_wrong = 0, 0
for i=1,batcher:num_batches() do
	local inputs, targets, sizes = unpack(batcher:next_batch())
	
	if opt.gpuid >= 0 then
		inputs = inputs:cuda()
		sizes = sizes:cuda()
	end

	local words = convert_inputs_to_words(inputs, sizes, idx_to_grapheme)
	local phonemes = convert_targets_to_phonemes(targets, idx_to_phoneme)

	local predictions = model:forward({ inputs, sizes }) -- batch_seq_length x batch x output_dim
	local cur_batchsize	= #targets											-- Last batch in epoch may be smaller than opt.batchsize
	predictions = predictions:view(-1, cur_batchsize, num_phonemes + 1)		-- seq_length x batch x output_dim
	
	-- Now use above to construct each predicted sequence
	local preds = convert_preds_to_phonemes(predictions, sizes, idx_to_phoneme)

 	local batch_num_correct, batch_num_wrong, batch_matches = unpack(check_wer(phonemes, preds))

 	-- Debugging: print original words, correct g2p, and predicted g2p 
 	for i=1,#targets do
 		print(batch_matches[i] .. ' ' .. words[i])
 		print(table.concat(phonemes[i], '-'))
 		print(table.concat(preds[i], '-'))
 	end

 	num_correct = num_correct + batch_num_correct
 	num_wrong = num_wrong + batch_num_wrong 
 	print(num_correct, num_wrong, num_wrong / (num_correct + num_wrong))
end
