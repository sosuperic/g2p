-- Load a network and evaluate it on the test set
require 'nngraph'
require 'rnn'

require 'MaskRNN'
require 'ReverseMaskRNN'

require 'utils.lua_utils'

require 'warp_ctc'

-- local gpuid = -1
-- if gpuid >= 0 then
-- 	inputs = inputs:cuda()
-- 	sizes = sizes:cuda()
-- end

local model = torch.load('models/2016_6_23___12_47_48/e10_0.7116.t7')

local dataset, grapheme_to_idx, phoneme_to_idx = unpack(require 'cmudict_data')
local num_graphemes = size_of_table(grapheme_to_idx)
local idx_to_grapheme = invert_table(grapheme_to_idx)
print(size_of_table(phoneme_to_idx))
local idx_to_phoneme = invert_table(phoneme_to_idx)

local batcher = require 'batcher'
local batchsize = 8		-- This has to be the same as the original model
local batcher = get_batcher(dataset.valid, batchsize, num_graphemes)

local get_decoder = require 'decoder'
local decoder = get_decoder(grapheme_to_idx, phoneme_to_idx)


-- tbl = {1: 16, 2: 12, 3: 20}
local function convert_idx_to_phoneme(tbl)
	local phonemes = {}
	for i, idx in ipairs(tbl) do
		table.insert(phonemes, idx_to_phoneme[idx])
	end
	return phonemes
end

for i=1,batcher:num_batches() do
	local inputs, targets, sizes = unpack(batcher:next_batch())
	local predictions = model:forward({ inputs, sizes }) -- seq_length x batch x output_dim


	print(predictions:size())

	for i=1,#targets do
		print(sizes[i], #targets[i])
	end


	local maxs, indices = torch.max(predictions, 3) -- max along 3rd dim

	local preds = {}
	for i=1,batchsize do
		local cur_pred = {}
		-- TOOD: how to know length of output sequence?
		-- ANSWER: MAX LENGTH IS ?, HAVE TO PASS IT THROUGH CTC TO GET PROB OF BLANK
		-- for j=1,sizes[i] do
		for j=1,2 do
			local phoneme = idx_to_phoneme[indices[i][j][1]]
			table.insert(cur_pred, phoneme)
		end
		table.insert(preds, cur_pred)

		-- Follow nnx
		local acts = torch.Tensor()
		acts:resizeAs(predictions):copy(predictions)
		acts:view(acts, acts:size(1) * acts:size(2), -1)
		local sizes_table = torch.totable(sizes)
		local grad_input = acts.new():resizeAs(acts):zero()	-- dummy
		local ctc_output = cpu_ctc(acts:float(), grad_input:float(), targets, sizes_table)

		print(ctc_output)

		-- print(acts)
		os.exit()



		local target = convert_idx_to_phoneme(targets[i])
		print(target)
		print(cur_pred)
		os.exit()
	end





	print(targets)
	print(predictions)
	os.exit()
end

