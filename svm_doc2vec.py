#coding:utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import time
from sklearn.svm import SVC
import gensim

VECTOR_DIR = 'vectors.bin'
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
TEST_SPLIT = 0.2
start_time = time.time()
train_docs = open('train_contents.txt', 'rb').read().split(b'\n')
train_labels = open('train_labels.txt', 'rb').read().split(b'\n')
test_docs = open('test_contents.txt', 'rb').read().split(b'\n')
test_labels = open('test_labels.txt', 'rb').read().split(b'\n')


def train_d2v_model():
    all_docs = train_docs + test_docs
    fout = open('all_contents.txt', 'wb')
    fout.write(b'\n'.join(all_docs))
    fout.close()
    sentences = gensim.models.doc2vec.TaggedLineDocument('all_contents.txt')
    model = gensim.models.Doc2Vec(sentences, vector_size=200, window=5, min_count=4)
    model.save('doc2vec.model')
    print('num of docs: ' + str(len(model.docvecs)))
        

if __name__ == '__main__':
    print('(1) training doc2vec model...')
    train_d2v_model()
    print('(2) load doc2vec model...')

    model = gensim.models.Doc2Vec.load('doc2vec.model')
    x_train = []
    x_test = []
    y_train = train_labels
    y_test = test_labels
    for idx, docvec in enumerate(model.docvecs):
        # print(idx, docvec)
        if idx < 17600:
            x_train.append(docvec)
        else:
            x_test.append(docvec)
        if idx == len(model.docvecs)-1:
            break
    print('train doc shape: '+str(len(x_train))+' , '+str(len(x_train[0])))
    print('test doc shape: '+str(len(x_test))+' , '+str(len(x_test[0])))
    print('train label shape: '+str(len(train_labels)))
    print('test label shape: '+str(len(test_labels)))
    print('(3) SVM...')

    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))
    end_time = time.time()
    print('运行时间：%ds' % (end_time-start_time))





        




