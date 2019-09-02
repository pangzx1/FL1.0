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
from federatedml.util import consts
from .test_sync_base import TestSyncBase


class DHKeyExchangeTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable):
        from federatedml.homo.sync import dh_keys_exchange_sync
        if role == consts.ARBITER:
            arbiter = dh_keys_exchange_sync.Arbiter()
            arbiter._register_dh_key_exchange(transfer_variable.dh_pubkey,
                                              transfer_variable.dh_ciphertext_host,
                                              transfer_variable.dh_ciphertext_guest,
                                              transfer_variable.dh_ciphertext_bc)
            return arbiter.key_exchange()
        elif role == consts.HOST:
            sync = dh_keys_exchange_sync.Host()
            sync._register_dh_key_exchange(transfer_variable.dh_pubkey,
                                           transfer_variable.dh_ciphertext_host,
                                           transfer_variable.dh_ciphertext_bc)
            sync.key_exchange("aaa")
        else:
            sync = dh_keys_exchange_sync.Host()
            sync._register_dh_key_exchange(transfer_variable.dh_pubkey,
                                           transfer_variable.dh_ciphertext_host,
                                           transfer_variable.dh_ciphertext_bc)
            sync.key_exchange("aaa")

    def runTest(self):
        num_hosts = 100
        results = self.run_results(num_hosts=num_hosts)
        arbiter = results[0]
        guest = results[1]
        hosts = results[2:]

        assert len(results) == num_hosts + 2
        assert guest in arbiter
        for host in hosts:
            assert host in arbiter
