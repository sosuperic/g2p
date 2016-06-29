-- This file produces batched inputs, targets, and sizes to pass to the network

-- Each call to next_batch() returns {inputs, targets, sizes}
-- inputs: Tensor of dim: batchsize x batch_seq_length x num_graphemes, where
	-- batch_seq_length is the length of the longest seq in the batch
-- targets: table of size batchsize
	-- Each item is a table that contained the phonemes mapped to their respective phonemes
	-- e.g. {1: {1: 11, 2: 20, 3: 29}} is a batch of batchsize=1, with the first word having 3 phonemes
-- sizes: Tensor of size batchsize
	-- Encodes the length of input (i.e. number of graphemes) so that we can apply a mask

function get_batcher(dataset_split, batchsize, num_graphemes)
	local batcher = {}
	local cur_idx = 0
	local cur_batch = 0

	function batcher:next_batch()
		local batch_seq_length = 0
		local batch_items = {}
		local sizes = {}
		local targets = {}
		for i=1,batchsize do
			local idx = cur_idx + i
			if idx > #dataset_split then
				cur_idx = 0
				break
			end

			local x = dataset_split[idx].x
			local y = dataset_split[idx].y
			local len = x:size(1)
			if len > batch_seq_length then
				batch_seq_length = len
			end
			table.insert(batch_items, x)
			table.insert(sizes, len)
			table.insert(targets, y)
		end

		-- Construct batched inputs: batchsize x batch_seq_length x num_graphemes
		local inputs = torch.zeros(#batch_items, batch_seq_length, num_graphemes)
		for i=1,#batch_items do
			inputs[{{i},{1,sizes[i]},{1,num_graphemes}}] = batch_items[i]
		end
		local sizes = torch.Tensor(sizes)		-- Convert table to tensor

		-- Reset if done with epoch
		-- else increment the cur batch index by 1, the current datum index by batchsize
		cur_batch = cur_batch + 1
		if cur_batch == self.num_batches() then
			cur_idx = 0
		else
			cur_idx = cur_idx + batchsize
		end

		return {inputs, targets, sizes}
	end

	-- Used to set for loop in training
	function batcher:num_batches()
		local n = math.ceil(#dataset_split / batchsize)
		return n
	end
	return batcher
end

return get_batcher