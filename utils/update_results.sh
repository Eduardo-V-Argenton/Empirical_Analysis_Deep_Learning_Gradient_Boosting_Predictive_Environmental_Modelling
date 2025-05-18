#!/bin/bash
#
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
  if [ "$branch" != "main" ]; then
    # Cria uma pasta temporária com o conteúdo dos logs dessa branch
    git checkout "$branch" -- results/"$branch"/
  fi
done
