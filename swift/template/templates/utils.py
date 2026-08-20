# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt

DEFAULT_SYSTEM = 'You are a helpful assistant.'


@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    auto_add_bos: bool = True


@dataclass
class EmptyTemplateMeta(TemplateMeta):
    # alignment: with use_accuracy_compatible (USE_ACCURACY_COMPATIBLE=1) the dummy template
    # adds no suffix token, so raw text -> token_ids matches PaddleFleet bit-for-bit. When the
    # switch is off, keep the base TemplateMeta default suffix ([['eos_token_id']]).
    suffix: Prompt = field(
        default_factory=lambda: [] if os.environ.get('USE_ACCURACY_COMPATIBLE', '0') == '1' else [['eos_token_id']])
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}'])
    chat_sep: Optional[Prompt] = None
    auto_add_bos: bool = True


register_template(ChatmlTemplateMeta(LLMTemplateType.chatml))
register_template(EmptyTemplateMeta(LLMTemplateType.dummy))
