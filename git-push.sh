#!/usr/bin/bash
# Use only remote : origin

REMOTE_REPO=$(git remote)
WORKING_BRANCH=$(git branch --show-current)

echo -e "\n-> git pull ..."
echo "-> remote : ${REMOTE_REPO}"
echo "-> branch : ${WORKING_BRANCH}"

git pull ${REMOTE_REPO} ${WORKING_BRANCH}

echo "-> git pull complete"
echo "-> git add ..."

git add --all

echo "-> git add complete"
echo "-> git status"

git status

echo "-> git commit ..."

if [ $# -eq 0 ]; then
	read -p "-> Enter commit message : " COMMIT_MESSAGE
else
	COMMIT_MESSAGE=${1}
fi

git commit -m "${COMMIT_MESSAGE}"

echo "-> git commit complete"
echo "-> git push ..."

git push ${REMOTE_REPO} HEAD:${WORKING_BRANCH}

if [ $? = 0 ]; then
	echo -e "-> git push complete\n"
else
	git reset --soft HEAD^
	echo "-> git push faild. Please check git error message."
fi
