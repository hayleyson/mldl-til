from transformers import AutoTokenizer
from collections import defaultdict

class BPETokenizer:
    """
    Byte-pair tokenizer implementation.
    Started off with huggingface tutorial (https://huggingface.co/course/chapter6/5?fw=pt)
    Added Byte-level BPE to the original tutorial code
    """

    def __init__(self, vocab_size = 50, byte_level = True):
        self.byte_level = byte_level
        self.word_freqs = defaultdict(int)
        self.vocab = [] 
        self.vocab_size = vocab_size
        self.vocab_freqs = defaultdict(int)
        self.merges = defaultdict(str)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    @staticmethod
    def str_to_bytes_str(string):
        return str(string.encode('utf-8').hex())

    @staticmethod
    def split_bytes_str(bytes_str):
        return [bytes_str[i:i+2] for i in range(0, len(bytes_str), 2)]

    @staticmethod
    def bytes_to_string(bytes_str):
        return bytes([int(bytes_str[i:i+2], 16) for i in range(0, len(bytes_str), 2)]).decode('utf-8')

    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def compute_vocab_freq(self, splits):
        vocab_cnt = defaultdict(int)
        for word in splits.keys():
            for token in splits[word]:
                if self.byte_level:
                    try:
                        vocab_cnt[BPETokenizer.bytes_to_string(token)] += self.word_freqs[word]
                    except:
                        vocab_cnt[token] += self.word_freqs[word]
                else:
                    vocab_cnt[token] += self.word_freqs[word]
        return vocab_cnt

    def train(self, corpus):
        
        for text in corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            if self.byte_level:
                new_words = [BPETokenizer.str_to_bytes_str(word) for word, offset in words_with_offsets]
            else:
                new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1
        # print('word frequencies')
        # print(self.word_freqs)
        
        if self.byte_level:
            # start with hexadecimal value of 0~255
            self.vocab = [str(hex(i))[2:] for i in range(0, 256)] + ["<|endoftext|>"]
        else:
            alphabet = []
            for word in self.word_freqs.keys():
                if self.byte_level:
                    word = BPETokenizer.split_bytes_str(word)
                for letter in word:
                    if letter not in alphabet:
                        alphabet.append(letter)
            alphabet.sort()

            self.vocab = ["<|endoftext|>"] + alphabet.copy()
            # print('initial vocab')
            # print(self.vocab)

        if self.byte_level:
            splits = {word: BPETokenizer.split_bytes_str(word) for word in self.word_freqs.keys()}
        else:
            splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        # print('initial splits')
        # print(splits)

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = self.merge_pair(*best_pair, splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])

        # print(f'vocab size: {len(self.vocab)}')

        self.vocab_freqs = self.compute_vocab_freq(splits)
        # print(f'vocab_frequencies: {sorted(self.vocab_freqs.items(), key=lambda x: x[1], reverse=True)}')


    def tokenize(self, text):
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        if self.byte_level:
            splits = [BPETokenizer.split_bytes_str(BPETokenizer.str_to_bytes_str(word)) for word in pre_tokenized_text]
            # print(splits)
        else:
            splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
        # print('before decoding:', sum(splits, []))
        for idx, split in enumerate(splits):
            split = [x if x in self.vocab else "<|unk|>" for x in split]
            if self.byte_level:
                for inner_idx, tok in enumerate(split):
                    try:
                        decoded = BPETokenizer.bytes_to_string(tok)
                        split[inner_idx] = decoded
                    except:
                        pass
            splits[idx] = split
        return sum(splits, [])


if __name__ == "__main__":

    # string = "手を洗いました"
    # bytes = string.encode('utf-8')
    # print('string [', string, '] to [', bytes.hex(), ']')
    # print('string [', ' ', '] to [', ' '.encode('utf-8').hex(), ']')

    corpus = [
        "手を洗いました",
        "你星期五做什麼",
        "vad gör du på fredag?",
        "വെള്ളിയാഴ്ച നിങ്ങൾ എന്താണ് ചെയ്യുന്നത്?",
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    corpus_test = ['お腹が空いたので何か食べてもいいですか？']

    # corpus = [str(x.encode('utf-8').hex()) for x in corpus]

    print('BBPE')
    tokenizer = BPETokenizer(byte_level = True)
    tokenizer.train(corpus)
    for text in corpus_test:
        res = tokenizer.tokenize(text)
        print(f'# tokens in the text: {len(res)}')
        print(f'tokenized text: {res}')

    print('BPE')
    tokenizer = BPETokenizer(byte_level = False)
    tokenizer.train(corpus)
    for text in corpus_test:
        res = tokenizer.tokenize(text)
        print(f'# tokens in the text: {len(res)}')
        print(f'tokenized text: {res}')

    # results
    # BBPE
    # # tokens in the text: 120
    # tokenized text: ['c3', 'a3', 'c4', 'a3', 'c4', 'ac', 'c3', 'a8', 'c4', 'a7', 'c2', 'b9', 'c3', 'a3', 'c4', 'a3', 'c4', 'ae', 'c3', 'a7', 'c2', 'a9', 'c2', 'ba', 'c3', 'a3', 'c4', 'a3', 'c4', 'a6', 'c3', 'a3', 'c4', 'a3', 'c5', '81', 'c3', 'a3', 'c4', 'a3', 'c2', 'ae', 'c3', 'a3', 'c4', 'a3', 'c2', 'a7', 'c3', 'a4', 'c2', 'bd', 'c4', 'b7', 'c3', 'a3', 'c4', 'a3', 'c4', 'ad', 'c3', 'a9', 'c2', 'a3', 'c5', '81', 'c3', 'a3', 'c4', 'a3', 'c2', 'b9', 'c3', 'a3', 'c4', 'a3', 'c2', 'a6', 'c3', 'a3', 'c4', 'a4', 'c4', 'a4', 'c3', 'a3', 'c4', 'a3', 'c4', 'a6', 'c3', 'a3', 'c4', 'a3', 'c4', 'a6', 'c3', 'a3', 'c4', 'a3', 'c2', 'a7', 'c3', 'a3', 'c4', 'a3', 'c4', 'bb', 'c3', 'a3', 'c4', 'a3', 'c4', 'ad', 'c3', 'af', 'c2', 'bc', 'c5', '81']
    # BPE
    # # tokens in the text: 60
    # tokenized text: ['ã', 'ģ', '<|unk|>', '<|unk|>', '<|unk|>', '<|unk|>', 'ã', 'ģ', '<|unk|>', '<|unk|>', '<|unk|>', 'º', 'ã', 'ģ', 'Ħ', 'ã', 'ģ', 'Ł', 'ã', 'ģ', '<|unk|>', 'ã', 'ģ', '<|unk|>', 'ä', '½', '<|unk|>', 'ã', 'ģ', 'ĭ', 'é', '£', 'Ł', 'ã', 'ģ', '<|unk|>', 'ã', 'ģ', '<|unk|>', 'ã', 'Ĥ', 'Ĥ', 'ã', 'ģ', 'Ħ', 'ã', 'ģ', 'Ħ', 'ã', 'ģ', '<|unk|>', 'ã', 'ģ', 'Ļ', 'ã', 'ģ', 'ĭ', '<|unk|>', '¼', 'Ł']
    # => BBPE longer sequence. BBPE no unk token. 
    # => can't figure out why japanese characters become weird after using pre-tokenizer.. #to-do