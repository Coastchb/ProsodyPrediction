# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from bosonnlp import BosonNLP
import os
import re
import sys
import codecs
import io
import random
import requests
from argparse import ArgumentParser as APR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

FINAL_PUNC = "。"


def segment_tag(text):
    nlp = BosonNLP('2DgGSC-8.33497.8yeNchBP6L9n')
    result = nlp.tag(text)

    words = result[0]['word']
    tags = result[0]['tag']

    assert len(words) == len(tags)

    return words, tags

def read_tags(file):
    with codecs.open(file, 'r', encoding='utf-8') as fr:
        lines = [line.strip().split("\t") for line in fr.readlines()]
    return [line[0] for line in lines], [lines[1] for line in lines]

def write_tags(words,tags,file):
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        fw.writelines('\n'.join([w + "\t" + t for w, t in zip(words, tags)]))

def get_sample(words, tags, text):
    ret = []
    word_index = 0
    for word, tag in zip(words, tags):
        boundary = '0'
        word_index += len(word)
        assert word_index <= len(text)
        if (word_index < len(text) and text[word_index] == '#'):
            boundary = text[word_index + 1]
            assert boundary in ['1', '2', '3', '4']
            word_index += 2

        # word   word_len    pos     boundary
        ret.append("{0:s}\t{1:d}\t{2:s}\t{3:s}".format(word, len(word), tag, boundary))
    return '\n'.join(ret)

def write_samples(samples, file):
    with codecs.open(file, 'w', encoding='utf-8') as fd:
        fd.writelines('\n\n'.join(samples))

def main():
    apr = APR(description = "segment and tag mandarin text to get train and test files for "
                            "prosody training and evaluation")

    apr.add_argument("src_file", help = "source file")
    apr.add_argument("train_file", help = "file for training prosody model")
    apr.add_argument("test_file", help = "file for evaluating prodody model")
    apr.add_argument("-t", dest="tag", action='store_true',  help="segment and tag text or not")
    apr.add_argument("--d", dest="tag_dir", default='data/tags', help = "location of tag files")
    apr.add_argument("--r", dest="ratio", type = float, default= 0.05, help = "test ratio (0,1) [0.05]")

    args = apr.parse_args()

    if(not os.path.exists(args.tag_dir)):
        os.makedirs(args.tag_dir)

    all_samples = []
    with codecs.open(args.src_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line_index in range(len(lines)):
            line = lines[line_index].strip()
            if(line == "" or line == "\n"):
                continue

            tag_file = os.path.join(args.tag_dir, "{0:07d}.tag".format(line_index))
            if(not args.tag):
                if(not os.path.exists(tag_file)):
                    print("Expect file {0:s} exists!".format(tag_file))
                    continue
                words, tags = read_tags(tag_file)
            else:
                pure_line = re.sub("#[1-4]", "", line)
                words, tags = segment_tag(pure_line)
                write_tags(words, tags, tag_file)
            sample = get_sample(words,tags,line)
            all_samples.append(sample)

    random.shuffle(all_samples)
    test_num = int(len(all_samples) * args.ratio)
    test_samples = all_samples[:test_num]
    train_sampels = all_samples[test_num:]

    write_samples(test_samples, args.test_file)
    write_samples(train_sampels, args.train_file)


main()
