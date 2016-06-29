-- Load a network and evaluate it on the test set
require 'model'
require 'model_google'

require 'utils.lua_utils'

require 'warp_ctc'

-------------------------------------------------------------------------------------------
-- Setup
------------------------------------------------------------------------------------------

-- local gpuid = -1
-- if gpuid >= 0 then
-- 	inputs = inputs:cuda()
-- 	sizes = sizes:cuda()
-- end

-- local model = torch.load('models/2016_6_28___16_53_4/e50_0.0046.t7')
-- local model = torch.load('models/2016_6_28___19_48_26/e6_0.7585.t7')
local model = torch.load('models/2016_6_28___21_47_24/e13_0.8734.t7')
model:evaluate()

local dataset, grapheme_to_idx, phoneme_to_idx = unpack(require 'cmudict_data')
local num_graphemes = size_of_table(grapheme_to_idx)
local idx_to_grapheme = invert_table(grapheme_to_idx)
-- print(size_of_table(phoneme_to_idx))

-- Add blank symbol for CTC
phoneme_to_idx['-'] = 0		-- TODO: off by one... this fixes it but hmm..
local idx_to_phoneme = invert_table(phoneme_to_idx)

-- print(idx_to_phoneme)
-- os.exit()
-- print(idx_to_phoneme)

local batcher = require 'batcher'
local batchsize = 8		-- This has to be the same as the original model
-- local batcher = get_batcher(dataset.valid, batchsize, num_graphemes)
local batcher = get_batcher(dataset.test, batchsize, num_graphemes)

local get_decoder = require 'decoder'
local decoder = get_decoder(grapheme_to_idx, phoneme_to_idx)

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

-- TODO: Off by one... not sure why this is the case. See Evaluator in CTC as well
	-- See TODO in this function, as well as TODO in 
	-- This is handled, but not elegantly. what's the root cause?
-- Decode predictions (tensor: 
-- If argmax is highest idx, this corresponds to blank variable in CTC, therefore skip
local function convert_preds_to_phonemes(predictions, sizes)
	local preds = {}

	-- Pluck out most likely phonemes by argmaxing over activations
	local maxs, indices = torch.max(predictions, 3)
	for i=1,batchsize do
		local cur_pred = {}
		local prev_phoneme = '-'				-- To decode CTC, can't have same token twice in a row
		for j=1, sizes[i] do
			local idx = indices[j][i][1]
			
			-- HERE's the FIX:
			idx = idx - 1

			local phoneme = idx_to_phoneme[idx]
			if phoneme ~= '-' and phoneme ~= prev_phoneme then
				table.insert(cur_pred, phoneme)
			end
			prev_phoneme = phoneme
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
	for i=1,#phonemes do
		if #phonemes[i] ~= #preds[i] then	-- Short circuit. Must have same number of phonemes obv
			num_wrong = num_wrong + 1
		else
			local wrong = false
			for j=1,#phonemes[i] do
				if phonemes[i][j] ~= preds[i][j] then
					wrong = true
				end
			end
			if wrong then
				num_wrong = num_wrong + 1
			else
				num_correct = num_correct + 1
			end
		end
	end
	return {num_correct, num_wrong}
end

------------------------------------------------------------------------------------------
-- Testing
------------------------------------------------------------------------------------------
local num_correct, num_wrong = 0, 0
for i=1,batcher:num_batches() do
	local inputs, targets, sizes = unpack(batcher:next_batch())
	local words = convert_inputs_to_words(inputs, sizes)
	local phonemes = convert_targets_to_phonemes(targets)

	local predictions = model:forward({ inputs, sizes }) -- batch_seq_length x batch x output_dim

	-- Now use above to construct each predicted sequence
	local preds = convert_preds_to_phonemes(predictions, sizes)

 	-- Debugging: print original words, correct g2p, and predicted g2p 
 	for i=1,batchsize do
 		print(words[i])
 		print(phonemes[i])
 		print(preds[i])
 	end

 	local batch_num_correct, batch_num_wrong = unpack(check_wer(phonemes, preds))
 	num_correct = num_correct + batch_num_correct
 	num_wrong = num_wrong + batch_num_wrong 
 	print(num_correct, num_wrong, num_wrong / (num_correct + num_wrong))
end

print(num_correct, num_wrong, num_wrong / (num_correct + num_wrong))
