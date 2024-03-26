#!/bin/bash
echo "1 : Download"
echo "2 : Upload"
echo "3 : Branch"
echo -n "Enter Mode: "
read MODE

if [  $MODE == 1 ]; then
    echo "Download Mode ..."

    git pull 

elif  [  $MODE == 2 ]; then
    echo "Download Upload ..."
    echo -n "Enter Commit: "
    read COMMIT

    git pull 
    git add .
    git commit -m $COMMIT
    git push

elif  [  $MODE == 3 ]; then
    echo "Download Upload ..."

    echo "1 : Create Branch"
    echo "2 : View ALL Branch"
    echo "3 : Checkout Branch"
    echo "4 : Delete Branch"

    echo -n "Enter Branch Mode: "
    read BRANCH_MODE

    if [  $BRANCH_MODE == 1 ]; then
        echo -n "Enter Branch NAME: "
        read BRANCH_NAME
        
        git branch $BRANCH_NAME

    elif  [  $BRANCH_MODE == 2 ]; then
        git branch --all

    elif  [  $BRANCH_MODE == 3 ]; then
        echo -n "Enter Branch NAME: "
        read BRANCH_NAME
        
        git checkout $BRANCH_NAME

    elif  [  $BRANCH_MODE == 4 ]; then
        echo -n "Enter Branch NAME: "
        read BRANCH_NAME
        
        git branch --delete $BRANCH_NAME

    else
        echo "please select os number, Branch Mode!!"
    fi

else
    echo "please select os number, mode run"

fi