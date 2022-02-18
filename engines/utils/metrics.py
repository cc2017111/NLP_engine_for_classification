import tensorflow as tf


def metrics(y_true, y_pred, configs):
    precision = -1.0
    recall = -1.0
    f1 = -1.0
    correct_label_num = 0
    total_label_num = len(y_true)

    measuring_metrics = configs.measuring_metrics

    for i in range(len(y_true)):
        y = tf.argmax(y_pred[i]).numpy()
        if y_true[i] == y:
            correct_label_num += 1

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    results = {}
    for measuring in measuring_metrics:
        results[measuring] = vars()[measuring]
    return results
