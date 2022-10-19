BASEPATH='./data/validation/02-model_training'
SRCPATH='./src/validation/02-model_training'
LANGUAGES='python  r'

declare -A command=( ["python"]="python3" ["r"]="Rscript")
declare -A suffix=( ["python"]="py" ["r"]="R")

for language in $LANGUAGES
do
    if [ -n "$(ls -A $BASEPATH/$language/*.csv 2>/dev/null)" ]
    then
        :
    else
        echo "generating data in $BASEPATH/$language/"
        ${command[$language]} $SRCPATH/$language/fit_model.${suffix[$language]} --num_seeds=150
    fi
    echo "data generated in $BASEPATH/$language/"
done