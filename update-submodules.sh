#!/bin/sh

git submodule update --init --recursive
cd libaddr2line
# Make sure the latest remote info is available
git fetch origin
# Get the current submodule commit (HEAD)
current_commit=$(git rev-parse HEAD)
# Get the latest commit on the remote branch (e.g., origin/main)
remote_commit=$(git rev-parse origin/main)
if [ "$current_commit" != "$remote_commit" ]; then
	echo "Submodule libaddr2line has updates available."

	current_branch=$(git branch --show-current)
	if [ "$current_branch" = "master" ]; then
		echo "Pushing to the master branch is not allowed. To proceed, please create a new branch, re-run 'update-submodules.sh', and open a merge into the main branch."
  		exit
	fi

	git pull origin main
	cd ..
	git add libaddr2line
	git commit -m "Updates libaddr2line submodule to latest commit"
	git push
else
	echo "Submodule libaddr2line is up to date."
	exit
fi
