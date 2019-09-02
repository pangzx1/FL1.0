#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import functools

import numpy as np

from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric, MetricType, MetricMeta
from federatedml.homo.procedure import aggregator, random_padding_cipher
from federatedml.model_base import ModelBase
from federatedml.model_selection import MiniBatch
from federatedml.optim import activation
from federatedml.statistic import data_overview
from federatedml.util.transfer_variable.homo_lr_transfer_variable import HomoLRTransferVariable

LOGGER = log_utils.getLogger()


class HomeDNNGuest(ModelBase):
    def __init__(self):
        super().__init__()
        self.aggregator = aggregator.Guest()
        self.cipher = random_padding_cipher.Guest()

        # self.gradient_operator = LogisticGradient()
        #
        # self.initializer = Initializer()
        # self.classes_ = [0, 1]
        #
        # self.evaluator = Evaluation()
        self.loss_history = []
        # self.is_converged = False
        # self.role = consts.GUEST

    def _init_model(self, params):
        self.party_weight = params.party_weight
        self.batch_size = params.batch_size
        self.max_iter = params.max_iter

        self.transfer_variable = HomoLRTransferVariable()
        self.aggregator.register_aggregator(self.transfer_variable)
        self.cipher.register_random_padding_cipher(self.transfer_variable)
        self.converge_flag_transfer = self.transfer_variable.converge_flag

    def fit(self, data_instances):
        if not self.need_run:
            return data_instances

        self.aggregator.initialize_aggregator(self.party_weight)

        self.__init_model(data_instances)

        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        for iter_num in range(self.max_iter):
            # mini-batch
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()
            total_loss = 0
            batch_num = 0

            # for batch_data in batch_data_generator:
            #     n = batch_data.count()
            #
            #     f = functools.partial(self.gradient_operator.compute,
            #                           coef=self.coef_,
            #                           intercept=self.intercept_,
            #                           fit_intercept=self.fit_intercept)
            #     grad_loss = batch_data.mapPartitions(f)
            #
            #     grad, loss = grad_loss.reduce(self.aggregator.aggregate_grad_loss)
            #
            #     grad /= n
            #     loss /= n
            #
            #     if self.updater is not None:
            #         loss_norm = self.updater.loss_norm(self.coef_)
            #         total_loss += (loss + loss_norm)
            #     delta_grad = self.optimizer.apply_gradients(grad)
            #
            #     self.update_model(delta_grad)
            #     batch_num += 1
            #
            # total_loss /= batch_num
            #
            # w = self.merge_model()
            if not self.need_one_vs_rest:
                metric_meta = MetricMeta(name='train',
                                         metric_type=MetricType.LOSS,
                                         extra_metas={
                                             "unit_name": "iters",
                                         })
                self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
                self.callback_metric(metric_name='loss',
                                     metric_namespace='train',
                                     metric_data=[Metric(iter_num, total_loss)])

            self.loss_history.append(total_loss)
            LOGGER.info("iter: {}, loss: {}".format(iter_num, total_loss))
            # send model
            w = self.aggregator.aggregate_and_get(w, suffix=(iter_num,))

            # send loss
            self.aggregator.send_loss(total_loss, iter_num)

            w = np.array(w)
            self.set_coef_(w)

            # recv converge flag
            converge_flag = self.converge_flag_transfer.get(0, suffix=(iter_num,))

            self.n_iter_ = iter_num
            LOGGER.debug("converge flag is :{}".format(converge_flag))

            if converge_flag:
                self.is_converged = True
                break

    def __init_model(self, data_instances):
        pass

    def predict(self, data_instances):

        if not self.need_run:
            return data_instances
        LOGGER.debug(
            "homo_lr guest need run predict, coef: {}, instercept: {}".format(len(self.coef_), self.intercept_))
        wx = self.compute_wx(data_instances, self.coef_, self.intercept_)
        pred_prob = wx.mapValues(lambda x: activation.sigmoid(x))
        pred_label = self.classified(pred_prob, self.predict_param.threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"1": x[1], "0": (1 - x[1])}])
        return predict_result
