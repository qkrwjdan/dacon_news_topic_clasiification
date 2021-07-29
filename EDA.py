def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)

	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0

	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words

	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words

# rd = [random_deletion(tokenized,0.2) for tokenized in train.tokenized]
# rd_augmentation = pd.DataFrame({'augmented' : rd, 'topic_idx': train.topic_idx})
# rd_augmentation["title"] = [" ".join(words) for words in rd_augmentation.augmented]

# rs = [random_swap(tokenized,2) for tokenized in train.tokenized]
# rs_augmentation = pd.DataFrame({'augmented' : rs, 'topic_idx': train.topic_idx})
# rs_augmentation["title"] = [" ".join(words) for words in rs_augmentation.augmented]
