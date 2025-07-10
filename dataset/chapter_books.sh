#!/bin/sh
dev_set=$1
data_dir=$2
sour_dir="$PWD"/$data_dir/$dev_set/book_txts
targ_dir="$PWD"/$data_dir/$dev_set/book_chapters

mkdir -p $targ_dir

for file in "$sour_dir"/*
do
  echo "chapterize file:$file"
  chapterize "$file"
  name=$(basename "$file" ".txt")
  if [ -d "$sour_dir"/../../../"$name"-chapters ]; then
    mv "$sour_dir"/../../../"$name"-chapters "$targ_dir"/.
  fi
done
