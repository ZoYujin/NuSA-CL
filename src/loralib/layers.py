# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from CLIP-LoRA: https://github.com/MaxZanella/CLIP-LoRA
# CLIP-LoRA is licensed under the GNU Affero General Public License v3.0.
# Its implementation is reconstructed from Microsoft loralib by Baijiong Lin.
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        fan_in_fan_out: bool = False,
        dropout_rate:float = 0,
        lora_m:bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            #self.scaling = self.lora_alpha / self.r
            self.scaling = self.lora_alpha/math.sqrt(self.r) # 
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}
        self.lora_m = lora_m
        # 추가: weight_residual을 저장할 변수 (초기엔 None)
        self.weight_residual = None

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            self.register_parameter(f'{lora_name}_lora_B', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
            # LoRA M matrix
            if self.lora_m and self.r > 0:            
                self.register_parameter(f'{lora_name}_lora_M', nn.Parameter(torch.zeros(self.r,self.r)))
                
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))
                if self.lora_m and self.r > 0:            
                    nn.init.zeros_(eval(f'self.{lora_name}_lora_M'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    # def merge_BA(self, param_name: str):
    #     lora_name = self.params_with_lora[param_name]
    #     return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))

    def merge_BA(self, param_name: str) -> torch.Tensor:
        lora_name = self.params_with_lora[param_name]
        A = getattr(self, f'{lora_name}_lora_A')
        B = getattr(self, f'{lora_name}_lora_B')
        if self.lora_m and self.r > 0:
            M = getattr(self, f'{lora_name}_lora_M')
            update = B @ M @ A
        else:
            update = B @ A
        # 원래 W와 같은 shape 으로
        W = getattr(self, param_name)
        return self.transpose(update).view_as(W)

    def merge_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            # 1) 기존 weight (frozen p) 대신 residual 을 베이스로 쓰고 있는지 확인
            if self.weight_residual is None:
                base = set_param(self, param_name, mode='get').detach()
            else:
                base = self.weight_residual.detach()
                # print(f"[MERGE DEBUG] {param_name} residual norm: {base.norm().item():.4f}")

            # 2) 추가되는 LoRA 파트
            additional = self.merge_BA(param_name) * self.scaling
            # print(f"[MERGE DEBUG] {param_name} additional norm: {additional.norm().item():.4f}")

            # 3) 실제 더해지는지 확인
            p_new = base + additional
            # (선택) p_new 에서 base 를 빼보면 additional 과 일치해야 함
            diff = (p_new - base - additional).abs().max().item()
            # print(f"[MERGE DEBUG] {param_name} merge residual check (should be 0): {diff:.3e}")

            # 4) 최종 설정
            set_param(self, param_name, param=p_new, mode='update')
            # print(f"[MERGE DEBUG] {param_name} merged ✓\n")

    # def merge_lora_param(self):
    #     for param_name, lora_name in self.params_with_lora.items():
    #         p = set_param(self, param_name, mode='get')
    #         # p_new = p + scaling * (B @ A) + weight_residual
    #         # 여기서 weight_residual이 있다면 이를 더해서 기존 weight의 효과를 보존하도록 합니다.
    #         additional = self.merge_BA(param_name) * self.scaling
    #         if self.weight_residual is not None:
    #             p_new = self.weight_residual.detach() + additional
    #         else:
    #             p_new = p.detach() + additional
    #         set_param(self, param_name, param=p_new, mode='update')
    #         print(param_name, "merged")

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling
    
    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling
    
    def lora_train(self, mode: bool = True):
        if mode:
            # if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                # self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True 


class LinearLoRA(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        lora_m:bool = False,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out, lora_m=lora_m)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def train(self, mode: bool = True):
        super().train(mode)     
        self.lora_train(mode)

    def lora_only_forward(self, x):
        if self.r == 0:
            return torch.zeros_like(F.linear(x, self.weight, self.bias))

        return torch.matmul(x,self.merge_BA('weight').transpose(0, 1)) * self.scaling 
        
    def forward(self, x: torch.Tensor, **kwargs):
        
        if self.dropout is None: # do as before
            if self.r > 0 and not self.merged:
                self.merge_lora_param()
                result = nn.Linear.forward(self, x, **kwargs)
                # self.sub_lora_data()
                return result
            else:
                return nn.Linear.forward(self, x, **kwargs)
            
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)

        if self.training and self.dropout.p > 0:
            x = self.dropout(x)
        
        if self.r > 0 and not self.merged:
            lora_adjustment = torch.matmul(x,self.merge_BA('weight').transpose(0, 1)) * self.scaling 
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result


class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(
            self,
            existing_mha: nn.MultiheadAttention,
            enable_lora: list = ['q', 'k', 'v', 'o'],
            r: int = 0, 
            lora_alpha: int = 1, 
            dropout_rate:float = 0.,
            lora_m:bool = False,
            **kwargs
        ):
        super().__init__()
        
        self.dropout = 0 # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        #self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=existing_mha.in_proj_bias is not None)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)
        self.lora_m = lora_m
        self.weight_residual = None
        self.encoder_type = None  # ← 태깅 공간 미리 확보
        self.layer_idx = None
        self.get_cur_feat = False
        # Initialize parameters
        with torch.no_grad():
            
            # Extract the existing weights and biases
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            # Initialize q_proj
            self.q_proj.weight.data.copy_(existing_weight[:self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[:self.embed_dim])

            # Initialize k_proj
            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim:2*self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim:2*self.embed_dim])

            # Initialize v_proj
            self.v_proj.weight.data.copy_(existing_weight[2*self.embed_dim:, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2*self.embed_dim:])

            # Initialize proj
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention
        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate, lora_m = lora_m)
        
        # Init qkv as a new lora linear layer 
        for item in enable_lora:
            if item == 'q':
                self.q_proj = LinearLoRA(self.q_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate, lora_m = lora_m)
            elif item == 'k':
                self.k_proj = LinearLoRA(self.k_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate, lora_m = lora_m)  
            elif item == 'v':
                self.v_proj = LinearLoRA(self.v_proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate,lora_m = lora_m)
            elif item == 'o':
                self.proj = LinearLoRA(self.proj,
                                         r=r,
                                         lora_alpha=lora_alpha,
                                         fan_in_fan_out=False,
                                         dropout_rate = dropout_rate, lora_m = lora_m)  
        
        # Get device and dtype from LoRA layers
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'proj']:
            proj_layer = getattr(self, proj_name)
            if isinstance(proj_layer, LinearLoRA):
                self.device = proj_layer.weight.device
                self.dtype = proj_layer.weight.dtype
                break
        else:
            raise ValueError("No LoRA-enabled projection layer found to determine device and dtype.")

        # Initialize GPM-related variables
        self.matrix = torch.zeros(self.embed_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        self.cur_matrix = torch.zeros(self.embed_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        self.n_matrix = 0
        self.n_cur_matrix = 0
        self.feature_list = None
        self.project_type = []  # 'remove' or 'retain' for each layer
        
    def forward_module(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        #self.lora_train(mode)  

    # def forward(self,
    #         query: torch.Tensor,
    #         key: torch.Tensor,
    #         value: torch.Tensor,
    #         **kwargs):
        

    #     return self.forward_module(query, key, value, **kwargs) 
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        get_feat: bool = False,
        get_cur_feat: bool = False,
        **kwargs
    ):
        get_cur_feat = self.get_cur_feat
        # print(f"[DEBUG] get_feat: {get_feat}, get_cur_feat: {get_cur_feat}")
        # 입력 정규화: (B, N, C) 형식으로 통일
        x = query if self.batch_first else query.transpose(0, 1)
        B, N, C = x.shape

        # Feature Matrix 누적 (전체 태스크 기준)
        if get_feat:
            with torch.no_grad():  # 메모리 효율성을 위해 gradient 계산 방지
                feat_mat = torch.bmm(x.permute(0, 2, 1), x).sum(dim=0).to(self.device)
                if self.n_matrix == 0:
                    self.matrix = feat_mat / (B * N)
                else:
                    self.matrix = (self.matrix * self.n_matrix + feat_mat) / (self.n_matrix + B * N)
                self.n_matrix += B * N
                # print(f"[DEBUG] Updated matrix, new n_matrix: {self.n_matrix}")
        # Current Feature Matrix 누적 (현재 태스크 기준)
        if get_cur_feat:
            with torch.no_grad():  # 메모리 효율성을 위해 gradient 계산 방지
                feat_mat = torch.bmm(x.permute(0, 2, 1), x).sum(dim=0).to(self.device)
                if self.n_cur_matrix == 0:
                    self.cur_matrix = feat_mat / (B * N)
                else:
                    self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + feat_mat) / (self.n_cur_matrix + B * N)
                self.n_cur_matrix += B * N
                # print(f"[DEBUG] Updated cur_matrix, new n_cur_matrix: {self.n_cur_matrix}")

        # 본래 attention 연산 수행
        return self.forward_module(query, key, value, **kwargs)

    def reset_cur_matrix(self):
        """현재 태스크의 activation matrix를 초기화"""
        self.cur_matrix.zero_()
        self.n_cur_matrix = 0

    def get_feature_list(self):
        """현재까지 수집된 feature list 반환"""
        return self.feature_list

    def set_feature_list(self, feature_list):
        """feature list 설정"""
        self.feature_list = feature_list

class DoubleLinearLoRA(nn.Module):
    """
    Null‐space + Principal‐space Dual‐LoRA with optional M‐matrix.
    scaling 계수는 여기서 직접 계산해 사용합니다.
    """
    def __init__(
        self,
        existing_linear: nn.Linear,
        r_null: int,
        r_princ: int,
        lora_alpha: float,
        fan_in_fan_out: bool = False,
        dropout_rate: float = 0.,
        lora_m: bool = False,
        ensemble_ratio: float = 0.5,
    ):
        super().__init__()

        # 1) 내부 서브모듈 생성
        self.null = LinearLoRA(
            existing_linear, r=r_null,
            lora_alpha=lora_alpha,
            fan_in_fan_out=fan_in_fan_out,
            dropout_rate=dropout_rate,
            lora_m=lora_m
        )
        self.princ = LinearLoRA(
            existing_linear, r=r_princ,
            lora_alpha=lora_alpha,
            fan_in_fan_out=fan_in_fan_out,
            dropout_rate=dropout_rate,
            lora_m=lora_m
        )

        # 2) Base weight/bias (frozen)
        self.weight = nn.Parameter(
            existing_linear.weight.data.clone(), requires_grad=False
        )
        self.bias = (
            nn.Parameter(existing_linear.bias.data.clone(), requires_grad=False)
            if existing_linear.bias is not None else None
        )

        # 3) 앙상블 비율
        self.ensemble_ratio = ensemble_ratio

        # 4) 직접 계산하는 scaling 계수
        self.scaling_null  = lora_alpha / math.sqrt(r_null) if r_null > 0 else 0.0
        self.scaling_princ = lora_alpha / math.sqrt(r_princ) if r_princ > 0 else 0.0

        # 5) dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # (2) Base projection
        y0 = F.linear(x, self.weight, self.bias)

        # (3) Null‐space adapter: B @ M @ A, then scaling_null
        Bn = self.null.w_lora_B      # (out, r_null)
        Mn = self.null.w_lora_M      # (r_null, r_null)
        An = self.null.w_lora_A      # (r_null, in)
        BA_null = Bn @ (Mn @ An)     # (out, in)
        y_null = F.linear(x, BA_null) * self.scaling_null

        # (4) Principal‐space adapter
        Bp = self.princ.w_lora_B
        Mp = self.princ.w_lora_M
        Ap = self.princ.w_lora_A
        BA_princ = Bp @ (Mp @ Ap)
        y_princ = F.linear(x, BA_princ) * self.scaling_princ

        # (5) Ensemble
        r = self.ensemble_ratio
        y_lora = r * y_null + (1 - r) * y_princ
        return y0 + F.normalize(y_lora, dim=-1)

    def merge_lora_param(self):
        """
        학습된 A,B,M 파라미터로 base weight를 직접 갱신합니다.
        forward와 동일한 B@M@A * scaling 로직을 사용합니다.
        """
        if self.merged:
            return

        base = self.weight.detach().clone()

        # Δ_null = (B@M@A) * scaling_null
        Bn, Mn, An = (
            self.null.w_lora_B.detach(),
            self.null.w_lora_M.detach(),
            self.null.w_lora_A.detach()
        )
        delta_null = (Bn @ (Mn @ An)) * self.scaling_null

        # Δ_princ
        Bp, Mp, Ap = (
            self.princ.w_lora_B.detach(),
            self.princ.w_lora_M.detach(),
            self.princ.w_lora_A.detach()
        )
        delta_princ = (Bp @ (Mp @ Ap)) * self.scaling_princ

        # Ensemble merge
        r = self.ensemble_ratio
        new_base = base + r * delta_null + (1 - r) * delta_princ

        # 덮어쓰기
        self.weight.data.copy_(new_base)
        self.merged = True
    def train(self, mode: bool = True):
        """
        모듈 전체와 내부 LoRA 모듈에 대해 train/eval 모드 전환
        """
        super().train(mode)
        # LinearLoRA 안의 LoRALayer에도 전달
        self.null.lora_train(mode)
        self.princ.lora_train(mode) 

class PlainMultiheadAttentionDualLoRA(nn.Module):
    """
    Dual‐LoRA를 적용한 MultiheadAttention.
    기존 MHA의 q,k,v,o projection을 각각 DoubleLinearLoRA로 감싼 뒤,
    앙상블 비율에 따라 null/principal 어댑터 출력을 섞어줍니다.
    """
    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: list,          # ['q','k','v','o'] 중 붙일 projection
        r_null: int,                # null‐space 차원
        r_princ: int,               # principal‐space 차원
        lora_alpha: float,          # scaling
        ensemble_ratio: float,      # null vs princ 앙상블 비율
        dropout_rate: float = 0.,   # 입력 드롭아웃
        lora_m: bool = False        # M‐매트릭스 옵션
    ):
        super().__init__()
        # ⏩ 기존 MHA 속성 복사
        self.embed_dim  = existing_mha.embed_dim
        self.num_heads  = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim    = existing_mha.head_dim
        self.dropout    = dropout_rate

        # scaled dot‐product attention 함수
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        # ⏩ 원본 qkv/o projection 레이어 생성
        existing_w = existing_mha.in_proj_weight.data
        existing_b = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=(existing_b is not None))
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=(existing_b is not None))
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=(existing_b is not None))
        self.proj   = nn.Linear(self.embed_dim, self.embed_dim, bias=(existing_mha.out_proj.bias is not None))

        # ⏩ weight/bias 복사
        self.q_proj.weight.data.copy_(existing_w[:self.embed_dim, :])
        self.k_proj.weight.data.copy_(existing_w[self.embed_dim:2*self.embed_dim, :])
        self.v_proj.weight.data.copy_(existing_w[2*self.embed_dim:, :])
        if existing_b is not None:
            self.q_proj.bias.data.copy_(existing_b[:self.embed_dim])
            self.k_proj.bias.data.copy_(existing_b[self.embed_dim:2*self.embed_dim])
            self.v_proj.bias.data.copy_(existing_b[2*self.embed_dim:])
        self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
        if existing_mha.out_proj.bias is not None:
            self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        # projection 어트리뷰트 매핑
        param_to_proj = {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
            'o': 'proj'       # ← 여기서 'o'는 실제로 self.proj에 대응
        }

        # enable_lora에 따라 해당 projection 레이어를 wrapping
        for name in enable_lora:
            proj_attr = param_to_proj.get(name)
            if proj_attr is None:
                raise ValueError(f"Unknown LoRA target: {name}")
            base = getattr(self, proj_attr)   # q_proj, k_proj, v_proj 또는 proj
            wrapped = DoubleLinearLoRA(
                base,
                r_null=r_null,
                r_princ=r_princ,
                lora_alpha=lora_alpha,
                fan_in_fan_out=False,
                dropout_rate=dropout_rate,
                lora_m=lora_m,
                ensemble_ratio=ensemble_ratio
            )
            setattr(self, proj_attr, wrapped)

        # GPM(Feature Accumulation)용 공간 초기화
        self.matrix     = torch.zeros(self.embed_dim, self.embed_dim)
        self.n_matrix   = 0
        self.cur_matrix = torch.zeros(self.embed_dim, self.embed_dim)
        self.n_cur_matrix = 0
        self.feature_list  = None

    def merge_lora_param(self):
        """
        학습된 null/princ LoRA를 앙상블 합산해
        base projection의 weight에 덮어씁니다.
        """
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'proj']:
            mod = getattr(self, proj_name)
            if isinstance(mod, DoubleLinearLoRA) and not mod.merged:
                mod.merge_lora_param()
        return

    def forward_module(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ):
        # 1) causal + mask 체크
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        # 2) batch_first 처리: 내부는 (seq, batch, dim) 형태여야 함
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        # 3) Q, K, V 계산
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 4) attn_mask 차원 정리
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.view(bsz * self.num_heads, tgt_len, src_len)
            else:
                raise RuntimeError(f"Unsupported attn_mask dim {attn_mask.dim()}")

        # 5) scaled dot‐product attention
        attn_output = self.scaled_dot_product_attention(
            q.view(bsz, self.num_heads, tgt_len, self.head_dim),
            k.view(bsz, self.num_heads, src_len, self.head_dim),
            v.view(bsz, self.num_heads, src_len, self.head_dim),
            attn_mask,
            self.dropout if self.training else 0.0,
            is_causal
        )  # (batch, heads, seq_len, head_dim)

        # 6) heads 차원 합치기: (batch, heads, seq_len, head_dim) → (seq_len, batch, embed_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  
        # 이제 shape = (tgt_len, bsz, num_heads, head_dim)
        attn_output = attn_output.view(tgt_len, bsz, self.num_heads * self.head_dim)
        # shape = (tgt_len, bsz, embed_dim)

        # 7) output projection
        out = self.proj(attn_output)  # DoubleLinearLoRA로 wrapping 되어 앙상블까지 자동 처리

        # 8) batch_first 복원
        if self.batch_first:
            out = out.transpose(1, 0)  # → (batch, seq_len, embed_dim)

        return out, None

    def train(self, mode: bool = True):
        """
        모듈 전체와 내부 LoRA 모듈에 대해 train/eval 모드 전환
        """
        super().train(mode)
        # DoubleLinearLoRA 안의 LinearLoRA에도 전달
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'proj']:
            mod = getattr(self, proj_name)
            if isinstance(mod, DoubleLinearLoRA):
                mod.train(mode)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        get_feat: bool = False,
        get_cur_feat: bool = False,
        **kwargs
    ):
        # LoRA GPM용 feature accumulation
        x = query if self.batch_first else query.transpose(1, 0)
        B, N, C = x.shape

        print(f"[DEBUG] get_feat: {get_feat}, get_cur_feat: {get_cur_feat}")

        if get_feat:
            feat = torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0)
            self.matrix = (self.matrix * self.n_matrix + feat) / (self.n_matrix + B * N)
            self.n_matrix += B * N
            print(f"[DEBUG] Updated matrix, new n_matrix: {self.n_matrix}")

        if get_cur_feat:
            feat = torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0)
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + feat) / (self.n_cur_matrix + B * N)
            self.n_cur_matrix += B * N
            print(f"[DEBUG] Updated cur_matrix, new n_cur_matrix: {self.n_cur_matrix}")

        # 실제 attention 연산
        return self.forward_module(query, key, value, **kwargs)

class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        nn.Embedding.reset_parameters(self)
        self.init_lora_param()

    def init_lora_param(self):
        if hasattr(self, 'w_lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.w_lora_A)
            nn.init.normal_(self.w_lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_train(mode)
        
    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Embedding.forward(self, x, **kwargs)
            # self.sub_lora_data()
            return result
        else:
            return nn.Embedding.forward(self, x, **kwargs)

class Conv1d(nn.Conv1d, LoRALayer):
    # LoRA implemented in a Conv1d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv1d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv1d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv1d.forward(self, x, **kwargs)
            # self.sub_lora_data()
            return result
        else:
            return nn.Conv1d.forward(self, x, **kwargs)

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            # self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)

class Conv3d(nn.Conv3d, LoRALayer):
    # LoRA implemented in a Conv3d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv3d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv3d.forward(self, x, **kwargs)
            # self.sub_lora_data()
            return result
        else:
            return nn.Conv3d.forward(self, x, **kwargs)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0 and any(enable_lora):
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        nn.Linear.reset_parameters(self)
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        delta_w = F.conv1d(
            eval(f'self.{lora_name}_lora_A').unsqueeze(0), 
            eval(f'self.{lora_name}_lora_B').unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return self.transpose(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_train(mode)        

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Linear.forward(self, x, **kwargs)
            # self.sub_lora_data()
            return result
        else:
            return nn.Linear.forward(self, x, **kwargs)