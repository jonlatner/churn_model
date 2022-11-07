pdflatex -interaction batchmode churn_model.tex
echo "PDF1 =" $?

pdflatex -interaction batchmode churn_model.tex
echo "PDF1 =" $?

rm *.aux
rm *.log
rm *.nav
rm *.out
rm *.snm
rm *.toc

