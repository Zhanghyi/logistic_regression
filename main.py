from util import *
from model import LogisticRegressionBinaryClassifier

if __name__ == '__main__':
    # 1. 加载数据
    X_train, y_train = load_data('train')
    X_dev, y_dev = load_data('dev')
    X_test = load_data('test')

    # 2. 利用训练集更新权重
    classifier = LogisticRegressionBinaryClassifier(iterations=5000, learning_rate=0.01)
    classifier.train(X_train, y_train)

    # 3. 对训练集进行分类
    y_train_pred = classifier.predict(X_train)
    print('prediction accuracy on train set is {}'.format(accuracy(y_train_pred, y_train)))

    # 4. 对验证集进行分类
    y_dev_pred = classifier.predict(X_dev)
    print('prediction accuracy on dev set is {}'.format(accuracy(y_dev_pred, y_dev)))

    # 5. 对测试集进行分类
    y_test = classifier.predict(X_test)
    save_data(X_test, y_test, 'testset.json')

    # 6. 保存loss曲线图
    save_loss_history(classifier.loss_history)
