(cd $1 && python abalone.py $2 $3)
(cd $1 && ./svm-train -t 0  -c 2 abalone.tr)
(cd $1 && ./svm-predict abalone.te abalone.tr.model $4)
