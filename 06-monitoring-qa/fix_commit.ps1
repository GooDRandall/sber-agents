# Скрипт для исправления коммита 23a6c31
git checkout 23a6c31
git checkout 2546829 -- docs/references/naive-rag.ipynb
git commit --amend --no-edit
git checkout main
git rebase 23a6c31

