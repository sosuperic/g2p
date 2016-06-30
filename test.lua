-- Load a network and evaluate it on the test set
require 'model'
require 'model_google'

require 'utils.lua_utils'

require 'warp_ctc'

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
local model = torch.load('models/2016_6_28___21_47_24/e13_0.8734.t7')
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
-- Utilities to convert batcher encoded inputs and targets back into words and phonemes
------------------------------------------------------------------------------------------
-- Convert input (word encoded as tensor) back to word so we can compare against predictions
-- batchsize x batch_seq_len x input_dim
local function convert_inputs_to_words(inputs, sizes)
	local words = {}
	for i=1,inputs:size(1) do 								-- 1 to batchsize
		local word = ''
		local one_hot_word = inputs[{{i},{},{}}]			-- one hot encoded: 1 x batch_seq_len x input_dim
		local maxs, indices = torch.max(one_hot_word, 3)
		-- print(sizes)
		-- print(i)
		for j=1,sizes[i] do
			local idx = indices[1][j][1]
			word = word .. idx_to_grapheme[idx]
		end
		table.insert(words, word)
	end
	return words
end

-- Convert target to phonemes so we can compare against predictions
-- tbl = {1: 16, 2: 12, 3: 20}
local function convert_targets_to_phonemes(targets)
	local phonemes = {}
	for i, encoded_phonemes in ipairs(targets) do
		local phoneme_seq = {}
		for j=1, #encoded_phonemes do
			local idx = encoded_phonemes[j]
			table.insert(phoneme_seq, idx_to_phoneme[idx])
		end
		table.insert(phonemes, phoneme_seq)
	end
	return phonemes
end


-- Decode predictions (tensor)
local function convert_preds_to_phonemes(predictions, sizes)
	local preds = {}

	-- Pluck out most likely phonemes by argmaxing over activations
	local maxs, indices = torch.max(predictions, 3)
	for i=1,indices:size(2) do					-- 1 to batchsize
		local cur_pred = {}
		local prev_phoneme = '-'				-- To decode CTC, can't have same token twice in a row
		for j=1,sizes[i] do
			local idx = indices[j][i][1]
			idx = idx - 1						-- warp-CTC indexing starts at 0 (where 0 is the blank character)
			local phoneme = idx_to_phoneme[idx]
			if phoneme ~= '-' and phoneme ~= prev_phoneme then
				table.insert(cur_pred, phoneme)
				prev_phoneme = phoneme
			end
		end
		table.insert(preds, cur_pred)
	end
	return preds
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

	local words = convert_inputs_to_words(inputs, sizes)
	local phonemes = convert_targets_to_phonemes(targets)

	local predictions = model:forward({ inputs, sizes }) -- batch_seq_length x batch x output_dim
	local cur_batchsize	= #targets											-- Last batch in epoch may be smaller than opt.batchsize
	predictions = predictions:view(-1, cur_batchsize, num_phonemes + 1)		-- seq_length x batch x output_dim
	
	-- Now use above to construct each predicted sequence
	local preds = convert_preds_to_phonemes(predictions, sizes)

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
