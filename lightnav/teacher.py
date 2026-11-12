"""Teacher probabilities over NaVILA's 10 answers: the parts that do not need NaVILA's code.

`scripts/extract_teacher_probs.py` runs these inside the NaVILA environment (transformers 4.37.2 with NaVILA's
patched Llama, legacy tuple caches). The tests run them on a tiny Llama with the student's transformers
(cache objects), so both cache forms are handled.
"""
import re
from typing import List, Optional, Sequence

import torch

from .questions import ANSWERS, from_low_level

# NaVILA's user turn, as in llava/data/dataset.py (LazyVLNCEDataset) and evaluation/vlnce_baselines/navila_trainer.py.
_QUESTION = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video "
    "of historical observations {images}, and current observation <image>\n. Your assigned task is: \"{instruction}\" "
    "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
    "degree, moving forward a certain distance, or stop if the task is completed."
)


def navila_question(instruction: str, num_frames: int = 8) -> str:
    """NaVILA's prompt for one decision. `instruction` is expected cleaned already (`data.clean_instruction`)."""
    return _QUESTION.format(images="<image>\n" * (num_frames - 1), instruction=instruction)


def answer_token_ids(tokenizer, answers: Sequence[str] = ANSWERS) -> List[List[int]]:
    """Each answer as the tokens that follow the assistant header, without special tokens."""
    return [tokenizer(a, add_special_tokens=False).input_ids for a in answers]


def _repeat(layers, n: int):
    return tuple(tuple(t.expand(n, *t.shape[1:]) for t in layer) for layer in layers)


def repeat_cache(past, n: int):
    """`n` copies of a batch-1 key/value cache along the batch axis, in the form the model returned it."""
    if isinstance(past, (tuple, list)):
        return _repeat(past, n)
    return type(past).from_legacy_cache(_repeat(past.to_legacy_cache(), n))


@torch.no_grad()
def score_answers(llm, prompt_embeds: torch.Tensor, answer_ids: Sequence[Sequence[int]],
                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """log P(answer | prompt) of every answer, summed over its tokens, as a float32 tensor [len(answer_ids)].

    One pass over the prompt (batch 1), then all answers at once on copies of its cache. The answers are padded on
    the right; with causal attention a token never sees the padding after it, so they need no mask.
    """
    out = llm(inputs_embeds=prompt_embeds, attention_mask=attention_mask, use_cache=True)
    first = torch.log_softmax(out.logits[0, -1].float(), -1)
    n, length = len(answer_ids), max(len(a) for a in answer_ids)
    dev = prompt_embeds.device
    ids = torch.zeros(n, length, dtype=torch.long, device=dev)
    lens = torch.tensor([len(a) for a in answer_ids], device=dev)
    for i, a in enumerate(answer_ids):
        ids[i, :len(a)] = torch.tensor(list(a), dtype=torch.long, device=dev)
    scores = first[ids[:, 0]].clone()
    if length > 1:
        cont = llm(input_ids=ids[:, :-1], past_key_values=repeat_cache(out.past_key_values, n), use_cache=False)
        logp = torch.log_softmax(cont.logits.float(), -1)
        tok = torch.gather(logp, 2, ids[:, 1:, None])[..., 0]
        valid = torch.arange(1, length, device=dev)[None] < lens[:, None]
        scores = scores + (tok * valid).sum(1)
    return scores.float()


_PATTERNS = [(0, re.compile(r"\bstop\b", re.I)), (1, re.compile(r"\bis move forward\b", re.I)),
             (2, re.compile(r"\bis turn left\b", re.I)), (3, re.compile(r"\bis turn right\b", re.I))]


def answer_from_text(text: str) -> Optional[int]:
    """The option index NaVILA's evaluation loop would act on for a generated text (navila_trainer.py), or None.

    Same rules: "stop" first, then the action; the number is read as cm or degrees, a value that is not a multiple of
    the step is snapped to the nearest of 25/50/75 cm or 15/30/45 degrees, and a missing number means one step.
    A multiple of the step beyond three steps (e.g. 100 cm) is not one of the 10 answers and gives None.
    """
    action = next((a for a, pat in _PATTERNS if pat.search(text)), None)
    if action is None:
        return None
    if action == 0:
        return 0
    unit, step, choices = ("cm", 25, (25, 50, 75)) if action == 1 else ("degree", 15, (15, 30, 45))
    word = {1: "move forward", 2: "turn left", 3: "turn right"}[action]
    m = re.search(r"%s (\d+) %s" % (word, unit), text)
    value = int(m.group(1)) if m else step
    if value % step:
        value = min(choices, key=lambda c: abs(c - value))
    steps = value // step
    return from_low_level(action, steps) if 1 <= steps <= 3 else None
