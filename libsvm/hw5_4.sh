(cd $1 && python income.py $2 $3)
(cd $1 && ./svm-train income.tr)
(cd $1 && ./svm-predict income.te income.tr.model $4)