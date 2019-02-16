from base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class CifarTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(CifarTrainer, self).__init__(sess, model, data, config, logger)
        self.config = config

    def train_epoch(self, val_x, val_y, train_x, train_y):
        total_size = train_x.shape[0]
        number_of_batches = int(total_size / self.config.batch_size)
        loop = tqdm(range(number_of_batches))
        losses = []
        accs = []
        for i in loop:
            mini_x = train_x[i * self.config.batch_size:(i + 1) * self.config.batch_size, :, :, :]
            mini_y = train_y[i * self.config.batch_size:(i + 1) * self.config.batch_size, :]
            loss, acc = self.train_step(mini_x, mini_y)
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print()
        print('Train Loss: ', loss)
        print('Train accuracy: %.1f%%' % acc)
        print()

        val_loss, val_acc = self.val_statistics(val_x, val_y)
        print()
        print('Validation Loss: ', np.mean(val_loss))
        print('Validation accuracy: %.1f%%' % val_acc)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
            'val_loss': np.mean(val_loss),
            'val_acc': val_acc
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self, mini_batch_train_x, mini_batch_train_y):
        _, loss = self.sess.run([self.model.train_step, self.model.cross_entropy],
                                feed_dict={self.model.x: mini_batch_train_x, self.model.y: mini_batch_train_y,
                                           self.model.keep_prob: self.config.keep_prop, self.model.training: True})
        acc = self.sess.run([self.model.accuracy],
                            feed_dict={self.model.x: mini_batch_train_x, self.model.y: mini_batch_train_y,
                                       self.model.keep_prob: self.config.keep_prop, self.model.training: False})
        return loss, acc

    def val_statistics(self, val_x, val_y):
        feed_dict = {self.model.x: val_x, self.model.y: val_y, self.model.keep_prob: self.config.keep_prop,
                     self.model.training: False}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
