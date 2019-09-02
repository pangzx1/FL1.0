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

from arch.api.utils.log_utils import LoggerFactory
from fate_flow.entity.metric import MetricMeta, MetricType, Metric
from federatedml.homo.procedure import aggregator, random_padding_cipher
from federatedml.model_base import ModelBase
from federatedml.optim import convergence
from federatedml.param.logistic_regression_param import LogisticParam
from federatedml.util.transfer_variable.homo_lr_transfer_variable import HomoLRTransferVariable

LOGGER = LoggerFactory.get_logger()


class HomeDNNArbiter(ModelBase):

    def __init__(self):
        super().__init__()
        self.aggregator = aggregator.Arbiter()
        self.cipher = random_padding_cipher.Arbiter()

        self.loss_history = []
        self.is_converged = False

    def _init_model(self, params: LogisticParam):
        self.transfer_variable = HomoLRTransferVariable()
        self.max_iter = params.max_iter
        self.eps = params.eps
        self.converge_func = convergence.AbsConverge(eps=self.eps)

        self.aggregator.register_aggregator(self.transfer_variable)
        self.cipher.register_random_padding_cipher(self.transfer_variable)
        self.converge_flag_transfer = self.transfer_variable.converge_flag

    def fit(self, data=None):
        if not self.need_run:
            return data

        self.aggregator.initialize_aggregator()
        self.cipher.exchange_secret_keys()

        for iter_num in range(self.max_iter):

            final_model = self.aggregator.aggregate_and_broadcast(suffix=(iter_num,))

            total_loss = self.aggregator.aggregate_loss(suffix=(iter_num,))

            self.loss_history.append(total_loss)

            if not self.need_one_vs_rest:
                metric_meta = MetricMeta(name='train',
                                         metric_type=MetricType.LOSS,
                                         extra_metas={
                                             "unit_name": "iters"
                                         })
                metric_name = self.get_metric_name('loss')
                self.callback_meta(metric_name=metric_name, metric_namespace='train', metric_meta=metric_meta)
                self.callback_metric(metric_name=metric_name,
                                     metric_namespace='train',
                                     metric_data=[Metric(iter_num, total_loss)])

            LOGGER.info("Iter: {}, loss: {}".format(iter_num, total_loss))

            converge_flag = self.converge_func.is_converge(total_loss)
            self.converge_flag_transfer.remote(converge_flag, suffix=(iter_num,))

            if converge_flag:
                self.is_converged = True
                break
        self.data_output = data
