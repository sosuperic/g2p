
---------------------------------------------------------------------------
-- Decoder
---------------------------------------------------------------------------
local function get_decoder(grapheme_to_idx, phoneme_to_idx)
	local decoder = {}

	-- prediction: seq_length x batch x output_dim
	function decoder:decode(prediction)
		for i=1,prediction:size(1) do
			for j=1,prediction:size(2) do
			end
		end
	end
	-- for i=1,size()
	-- decoder
	return decoder
end

return get_decoder