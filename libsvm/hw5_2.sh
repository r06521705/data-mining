(cd $1 && ./svm-train -t 0 -c 1.25  $2)
(cd $1 && ./svm-predict $3 $2.model $4)