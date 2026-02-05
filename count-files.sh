#!/usr/bin/env bash

DIR="$1"

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

find "$DIR" -mindepth 1 -maxdepth 1 -print0 |
while IFS= read -r -d '' item; do
  if [ -d "$item" ]; then
    count=$(find "$item" -type f | wc -l)
    echo "$(basename "$item")/  $count files"
  else
    echo "$(basename "$item")"
  fi
done

