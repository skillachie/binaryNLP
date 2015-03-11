
#Get model
curl -O http://island2.cs.columbia.edu:8013/model.tgz
echo 'Done downloading model...'

#Extract model to directory
echo 'Extracting model...'
tar -xf model.tgz -C ../model2 --strip-components=1
echo 'Done extracting model...'

#Extract the data directory
echo 'Extracting sample date.. please wait.'
tar -xf ../data.tgz -C ../


#TODO install virtualenv and any python dependencies
if [ ! -d $HOME/py_env ]; then
	pip install virtualenv
	virtualenv py_env
fi

source $HOME/py_env/bin/activate
pip install -U numpy scipy scikit-learn
pip install pandas

echo 'To test functionality run do the following'
echo 'source $HOME/py_env/bin/activate'
echo 'python generate_binary_features.py '
echo 'This should create a csv called '
