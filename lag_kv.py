# Copyright 2025 China Merchants Bank. All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://mit-license.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.cache_utils import DynamicCache
import torch


class LagKV(DynamicCache):

    def __init__(self, _distributed_cache_data = None,
                 ratio: float = 0.5,
                 sink_size: int = 16,
                 lag_size: int = 1024,
                 score_v_ratio: float = 1.0,
                 skip_layer_idx: list = [],
                 use_then_compress: bool = True, # important
                ):
        super().__init__(_distributed_cache_data)
        self.ratio = ratio
        self.sink_size = sink_size
        self.lag_size = lag_size
        self.score_v_ratio = score_v_ratio
        self.skip_layer_idx = skip_layer_idx
        self.use_then_compress = use_then_compress
        self._compressed_len = []
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None,
    ):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                    self._compressed_len.append(self.sink_size)
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self._compressed_len.append(self.sink_size)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
            if layer_idx not in self.skip_layer_idx:
                return self._compress_kv_by_lag(layer_idx)
                
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _get_states_score(self, base_len, in_size, end_idx, value):
        target_v = value[:, :, base_len:end_idx]
        target_v = target_v.view(in_size[0], in_size[1], -1, self.lag_size, in_size[-1])
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]

        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)

        score = ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
        
        return score        
    
    def _compress_algo(self, layer_idx, base_len):
        in_size = self.key_cache[layer_idx].size()
        end_idx = base_len + ((in_size[-2] - base_len) // self.lag_size) * self.lag_size
        key_score = self._get_states_score(base_len, in_size, end_idx, self.key_cache[layer_idx])
        value_score = self._get_states_score(base_len, in_size, end_idx, self.value_cache[layer_idx])
        score = key_score + value_score * self.score_v_ratio
        # you may need to sort the index for some cases
        selected_idx = torch.topk(score, int(self.ratio * self.lag_size), dim=-1).indices
        for i in range(1, selected_idx.size()[2]):
            selected_idx[:, :, i] += i * self.lag_size
        selected_idx = selected_idx.reshape(in_size[0], in_size[1], -1).unsqueeze(-1).expand(
                                                            -1, -1, -1, in_size[-1])
        new_base_len = base_len + selected_idx.size()[-2]
        # alwarys keep the last window
        tail_len = self.lag_size + in_size[-2] - end_idx
        
        def modify_kv(value):
            selected_value = torch.gather(value[:, :, base_len:end_idx], -2, selected_idx)
            value[:, :, base_len:new_base_len] = selected_value
            # move value forward, clone for the case of overlap
            value[:, :, new_base_len:(new_base_len + tail_len)] = value[:, :, -tail_len:].clone()
            value = value[:, :, :(new_base_len + tail_len)]
            return value
            
        self.key_cache[layer_idx] = modify_kv(self.key_cache[layer_idx])
        self.value_cache[layer_idx] = modify_kv(self.value_cache[layer_idx])
        self._compressed_len[layer_idx] = new_base_len
    
    def _compress_kv_by_lag(self, layer_idx):
        kv_size = self.key_cache[layer_idx].size()
        base_len = self._compressed_len[layer_idx]
        tmp_key, tmp_value = self.key_cache[layer_idx], self.value_cache[layer_idx]
        if kv_size[-2] >= base_len + 2*self.lag_size:
            if self.use_then_compress:
                tmp_key, tmp_value = self.key_cache[layer_idx].clone(), self.value_cache[layer_idx].clone()
            self._compress_algo(layer_idx, base_len)
            if not self.use_then_compress:
                tmp_key, tmp_value = self.key_cache[layer_idx], self.value_cache[layer_idx]
        return tmp_key, tmp_value
        