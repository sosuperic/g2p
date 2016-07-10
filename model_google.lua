-- LSTM model for g2p
require 'nngraph'
require 'rnn'

require 'MaskRNN'
require 'ReverseMaskRNN'

-- nngraph.setDebug(true)

-- Convert unidirectional into bidirectional
-- We do this instead of simply using say cudnn.BLSTM in order to mask
-- NOTE: batch should be second dimension
local function convert_to_masked_bidirectional(rnn, seq_lengths, rnn_module)
	local fwd = nn.MaskRNN(rnn_module:clone())({rnn, seq_lengths})
	local bwd = nn.ReverseMaskRNN(rnn_module:clone())({rnn, seq_lengths})
	local bi = nn.CAddTable()({fwd, bwd})
	return bi
end

local function convert_to_masked_unidirectional(rnn, seq_lengths, rnn_module)
	local fwd = nn.MaskRNN(rnn_module:clone())({rnn, seq_lengths})
	return fwd
end

-- Dependent on if using GPU; could also be RNN/LSTM/GRU
-- Expects seq_length x batch x input_dim
local function get_rnn_module(use_cudnn, rnn_input_size, rnn_hidden_size)
	local rnn_module = nil
	if use_cudnn then
		rnn_module = cudnn.LSTM(rnn_input_size, rnn_hidden_size, 1)
	else
		rnn_module = nn.SeqLSTM(rnn_input_size, rnn_hidden_size)
	end
	return rnn_module
end

local function g2p_model(use_cudnn, num_graphemes, num_phonemes)
	-- Rnn parameters
	local rnn_input_size = num_graphemes
	local rnn_num_hidden_layers = 2

	-- seq_lengths need to be passed through network so that CTC can access it
	-- It is also used for masking
	-- We don't do anything with it. Therefore just identity module	
	local seq_lengths = nn.Identity()()

	-- feat is module that takes input and converts it to feature vector
	-- This could be linear layers (or conv if working with spatial)
	-- Right now, just pass through one hot encodings
	local input = nn.Identity()()
	local feat = nn.Sequential()
	feat:add(nn.Transpose({1,2})) -- batch x seq_length x input_dim --> seq_length x batch x input_dim
	-- This is required by MaskRNN... seems odd
	feat:add(nn.View(-1, rnn_input_size)) -- (seq_length x batch) x features

	-- Rnn part of network takes our feature vector
	-- We get the appropriate rnn module (e.g. cudnn.LSTM vs. nn.SeqLSTM)
	-- First hidden layer has parallel tracks
	local rnn = nn.Identity()({ feat(input) })
	local rnn_module = get_rnn_module(use_cudnn, rnn_input_size, 512)
	local rnn_left = convert_to_masked_unidirectional(rnn, seq_lengths, rnn_module)
	local rnn_right = convert_to_masked_bidirectional(rnn, seq_lengths, rnn_module)

	-- Parallel tracks joined, then fully connected to second rnn 
	local jt = nn.JoinTable(2)({rnn_left, rnn_right})	-- (seq x batch) x (feat_left + feat_right)
	local fc = nn.Linear(1024, 128)(jt)
	local dr = nn.Dropout(0.5)(fc)
	rnn_module = get_rnn_module(use_cudnn, 128, 128)
	local rnn2 = convert_to_masked_unidirectional(dr, seq_lengths, rnn_module)

	-- Add linear layer to output of rnn
	local post_rnn = nn.Sequential()
	post_rnn:add(nn.Linear(128, num_phonemes + 1))	-- + 1 for the BLANK character in CTC

	-- Glue together module
	local model = nn.gModule({input, seq_lengths}, { post_rnn(rnn2) })
	return model 
end

return g2p_model
