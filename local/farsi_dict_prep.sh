#! /usr/bin/env bash

if [ ! -d data/local/dict ]; then
	mkdir -p data/local/dict
fi

# After applying the farsi_preprocessing script the dictionary needs a few beauty operations
# namingly. removing the EOS and spoken noise tokens, removing all double spaces, and spaces at the end of a line
#cp /group/project/summa/svandiek/farsi/fa_grapheme_dict_prep.txt data/local/dict/lexicon_nosil.txt
#cp /disk/scratch1/svandiek/persian/lm/new_farsi_grapheme_dict.txt data/local/dict/lexicon_nosil.txt
#cp /disk/data1/svandiek/persian/data/new_grapheme_dict_500k.txt data/local/dict/lexicon_nosil.txt
cp farsi_asr_grapheme_dict.txt data/local/dict/lexicon_nosil.txt

(printf '!SIL\tsil\n[EN]\tspn\n[UNK]\tspn\n[EH]\tspn\n[SN]\tspn\n';) \
    | cat - data/local/dict/lexicon_nosil.txt \
> data/local/dict/lexicon.txt;

# taken from egs/gp/s5/local/gp_dict_prep.sh
{ echo sil; echo spn; } > data/local/dict/silence_phones.txt
echo sil > data/local/dict/optional_silence.txt


# The "tail -n +2" is a bit of a hack as other wise there will be an empty line in the file, which Kaldi doesn't like
cut -f2- data/local/dict/lexicon_nosil.txt | tr ' ' '\n' | sort -u | tail -n +2 \
> data/local/dict/nonsilence_phones.txt

( tr '\n' ' ' < data/local/dict/silence_phones.txt; echo;
    tr '\n' ' ' < data/local/dict/nonsilence_phones.txt; echo;
) > data/local/dict/extra_questions.txt

echo "Done with dict prep"