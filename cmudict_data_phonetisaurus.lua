-- Main differences with phonetisaurus:
-- word and phonemes are tab delimited
-- words are uppercase

-- Return dataset using CMUDict's pronunciation dictionary

-- Dataset is table consisting of three entries: train, valid, test
-- Each split (e.g. train) is table of data points
-- Each split is sorted by the length of the grapheme sequence
-- Eacha data point contains:
	-- 1) word: the original word
	-- 2) phonemes: table of phonemes
	-- 3) x: word encoded as tensor (num_graphemes_in_word x num_graphemes)
	-- 4) y: the phonemes encoded as table (each phoneme indexed according to phoneme_to_idx)

-- Link: http://www.speech.cs.cmu.edu/cgi-bin/cmudict?in=dictionary&stress=-s

require 'utils.lua_utils'

local PHONEME_PATH = 'cmudict/cmudict.phones'
local TRAIN_PATH = 'cmudict/phonetisaurus/cmudict.dic.train'
local TEST_PATH = 'cmudict/phonetisaurus/cmudict.dic.test'

------------------------------------------------------------------------
-- Get set of phonemes
------------------------------------------------------------------------
function get_phonemes()
	local phonemes = lines_from(PHONEME_PATH)
	local phoneme_to_idx = {}
	for i=1,#phonemes do
		local p = split(phonemes[i], '\t')[1]
		phonemes[i] = p
		phoneme_to_idx[p] = i
	end
	return {phonemes, phoneme_to_idx}
end

------------------------------------------------------------------------
-- Get dictionary (words to phonemes)
------------------------------------------------------------------------
-- Remove words with more than one pronunciiation 
	-- Seems like this is what is done by: http://www.ee.columbia.edu/~stanchen/papers/a014d.pdf
	-- (Cited by the reference paper)
-- Remove lexical stress markers (0, 1, 2) after certain phonemes
-- Create set of graphemes while we're at it
-- Remove words with comments (#). Only 8 -- these include french, name, old, abbreviation
-- Remove words that contain non-letters (punctuation and numbers)
function remove_lexical_stress(phonemes)
	for i,p in ipairs(phonemes) do
		p = p:gsub('%d', '')
		phonemes[i] = p
	end
	return phonemes
end

function split_line_into_word_and_phonemes(dict, i)
	-- local line = split(dict[i], ' ')
	local line = split(dict[i], '\t')
	local word = line[1]:lower()
	local phonemes = split(line[2], ' ')
	phonemes = remove_lexical_stress(phonemes)
	return {word, phonemes}
end

-- Check if the word on this line has multiple pronunciations
function word_is_invalid(dict, i)
	-- Line has comment, is special case
	if dict[i]:find('#') then
		return true
	-- Line has a parentheses, e.g. abalone(2)
	elseif dict[i]:find('%(') then
		return true
	-- This is the first instance of the word, next line has parentheses
	-- First check if we're at the end
	elseif i == #dict then
		return false
	-- Ok, now check next line
	elseif dict[i+1]:find('%(') then
		return true
	-- Ok, only one pronunciation
	else
		return false
	end
end

function add_to_grapheme_set(graphemes, word)
	-- for i=1,#word do
	-- 	graphemes[word:sub(i,i)] = true
	-- end
	for g in word:gmatch"." do
		graphemes[g] = true
	end
	return graphemes
end

function convert_grapheme_set_to_lookup(graphemes)
	grapheme_to_idx = {}
	local i = 1
	for g,_ in pairs(graphemes) do
		grapheme_to_idx[g] = i
		i = i + 1
	end
	return grapheme_to_idx
end


function get_data_and_graphemes(path)
	local dict = lines_from(path)
	local data = {}
	local graphemes = {}
	for i=1,#dict do
		if not word_is_invalid(dict, i) then
			local word, phonemes = unpack(split_line_into_word_and_phonemes(dict, i))
			-- Ignore words that include non-letters (uppercase %A matches inverse set)
			if not word:match('%A') then
				if #word >= #phonemes then -- CTC doesn't work when len of target sequence > len of input seq
					graphemes = add_to_grapheme_set(graphemes, word)
					table.insert(data, {word=word, phonemes=phonemes})
				end
			end
		end
	end
	local grapheme_to_idx = convert_grapheme_set_to_lookup(graphemes)
	return {data, grapheme_to_idx}
end


------------------------------------------------------------------------
-- Prepare data: encode graphemes and phonemes
------------------------------------------------------------------------

-- Takes string of graphemes (i.e. a word)
-- Returns tensor of dimension (num_graphemes_in_word x num_graphemes)
function one_hot_encode_word(word, grapheme_to_idx)
	local x = torch.zeros(#word, size_of_table(grapheme_to_idx))
	local i = 1
	for g in word:gmatch"." do
		-- print(i, grapheme_to_idx[g])
		x[i][grapheme_to_idx[g]] = 1
		i = i + 1
	end
	return x
end

-- Takes table, each entry is a phoneme
-- Return table of size num_phonemes_in_word
	-- e.g. {1, 4, 8, 3}
function encode_phonemes(phonemes, phoneme_to_idx)
	local y = {}
	for i,p in ipairs(phonemes) do
		table.insert(y, phoneme_to_idx[p])
	end
	return y
end

function encode_data(data, grapheme_to_idx, phoneme_to_idx)
	for i=1,#data do
		local word = data[i].word
		-- if string.find(word, '#') then
		-- 	print(word)
		-- end
		local phonemes = data[i].phonemes
		data[i]['x'] = one_hot_encode_word(word, grapheme_to_idx)
		data[i]['y'] = encode_phonemes(phonemes, phoneme_to_idx)
	end
	return data
end


------------------------------------------------------------------------
-- Prepare data: training/valid/test splits
------------------------------------------------------------------------
function shuffle_table(t)
	for i=#t,2,-1 do	-- Go backwards
		local r = math.random(i)		-- random number between 1 and i
		t[i], t[r] = t[r], t[i]			-- swap random item to position i
	end
	return t
end

-- Table will be in increasing seq length
function sort_by_seq_length(dataset_split)
	local function sorter(a, b)
		if (a.x:size(1) < b.x:size(1)) then return true else return false end
	end
	table.sort(dataset_split, sorter)
	return dataset_split
end

function split_data(data)
	data = shuffle_table(data)
	local valid = sort_by_seq_length(subrange(data, 1, NUM_VALID))
	local test = sort_by_seq_length(subrange(data, NUM_VALID+1, NUM_VALID+NUM_TEST))
	local train = sort_by_seq_length(subrange(data, NUM_VALID+NUM_TEST+1, #data))
	print(#data, #train, #valid, #test)
	return {train=train, valid=valid, test=test}
end

function count_phonemes(split)
	local counts = defaultdict(0)
	for i=1,#split do
		for j=1,#split[i].phonemes do
			p = split[i].phonemes[j]
			counts[p] = counts[p] + 1
		end
	end
	for p,c in pairs(counts) do
		counts[p] = c / #split
	end
	return counts
end

function check_phonetic_balance(dataset)
	local tr_counts = count_phonemes(dataset.train)
	local va_counts = count_phonemes(dataset.valid)
	local te_counts = count_phonemes(dataset.test)

	local counts = defaultdict('')
	for p,c in pairs(tr_counts) do
		counts[p] = tonumber(c)
	end
	for p,c in pairs(va_counts) do
		counts[p] = counts[p] .. ' ' .. tonumber(c)
	end
	for p,c in pairs(te_counts) do
		counts[p] = counts[p]  .. '' .. tonumber(c)
	end
	print(counts)
end

------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------
local phonemes, phoneme_to_idx = unpack(get_phonemes())
local train_data, grapheme_to_idx = unpack(get_data_and_graphemes(TRAIN_PATH))
local test_data, _ = unpack(get_data_and_graphemes(TEST_PATH))
train_data = encode_data(train_data, grapheme_to_idx, phoneme_to_idx)
test_data = encode_data(test_data, grapheme_to_idx, phoneme_to_idx)
local train = sort_by_seq_length(train_data)
local test = sort_by_seq_length(test_data)
-- print(train)
-- os.exit()
local dataset = {train=train, valid=test}
-- local dataset = split_data(data)
-- check_phonetic_balance(dataset)

-- torch.save('dataset_and_mappings.t7', {dataset, grapheme_to_idx, phoneme_to_idx})
-- return torch.load('dataset_and_mappings.t7')

return {dataset, grapheme_to_idx, phoneme_to_idx}

