export STANFORD_CORENLP_PATH="/home/ubuntu/stanfordcorenlp/stanford-corenlp-4.4.0"
export STANFORD_CORENLP_PORT=10086

cd $STANFORD_CORENLP_PATH && java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port $STANFORD_CORENLP_PORT -timeout 15000
