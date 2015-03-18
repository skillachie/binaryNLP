
#Get model
curl -O http://island1.cs.columbia.edu:8013/model.tgz
echo 'Done downloading model...'

#Extract model to directory
echo 'Extracting model...'
tar -xf model.tgz -C ../model --strip-components=1
echo 'Done extracting model...'

#Extract the data directory
echo 'Extracting sample date.. please wait.'
tar -xf ../data.tgz -C ../


#TODO install virtualenv and any python dependencies
if [ ! -d $HOME/py_env ]; then
	pip install virtualenv
	virtualenv py_env
fi

source py_env/bin/activate
pip install -U numpy scipy scikit-learn
pip install pandas


echo '*********INSTRUCTIONS***********'

echo 'To test functionality run do the following'
echo '1.source setup/py_env/bin/activate'
echo '2.python generate_binary_features.py '
echo '3.This should create a csv called nlp_category_binary_features.csv '
