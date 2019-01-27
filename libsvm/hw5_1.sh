(cd $1 &&  ./svm-scale $2 && ./svm-train -t 0 $2)
(cd $1 && ./svm-predict $3 $2.model $4)