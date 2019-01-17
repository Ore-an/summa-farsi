#!/bin/bash
# Copyright 2018 Ondrej Klejch
#			2019 Simon Vandieken

mkdir -p tmp
rm -rf tmp/*

sort -k 1,1 -k 4,4n data/summa/stm_after_plural_fix | \
   cut -f 1,6- -d ' ' | \
   awk -F $' ' '{if($2) {$1==""; print $0 >> "tmp/"$1".txt"}}'

for f in tmp/*.txt; do
   name=`basename $f | sed 's/.txt$//'`
   awk -v str="$name" '{str = str" "$0} END {print str}' $f >> data/summa/reference_after_plural_fix
done

rm -rf tmp

# insert this between sort and cut to work with Ondrejs utterance names
#sed 's/src-SUMMA_ses-//;s/\([^+]*\)+\([^+]*\)+\([^+]*\)+\([^ ]*\) /\1+\2+\3+\4\t/' |