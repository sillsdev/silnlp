"""Extend a pretrained tokenizer's vocabulary using the training corpus.

Two kinds of token get added. Characters the tokenizer does not know have to be added outright,
or they tokenize to ``<unk>`` and the text they came from is unrecoverable. On top of those, a BPE
tokenizer can be given extra merge rules learned from the corpus.

The merge rules are learned *after* the base tokenizer has segmented the corpus, over the sequences
of tokens it produced, and are appended to the **end** of the merge list so they rank below every
base rule. A BPE model applies merges in rank order, so the base tokenizer always runs to its fixed
point before any learned rule fires. The two phases separate cleanly: every learned merge produces a
token absent from the base vocabulary, so no base merge can become applicable once one has fired.

Each learned merge contributes exactly one new vocabulary entry, which is what makes the requested
number of tokens the number actually added.

Finding missing characters applies to any tokenizer; the functions named ``bpe`` or dealing in
merges are specific to BPE models, which Unigram models have no equivalent of.
"""

import json
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from tokenizers import Tokenizer

# A word (as split off by the pre-tokenizer) mapped to how often it occurs in the corpus.
WordCounts = Counter  # Counter[str]
# A word, as the sequence of base tokens it segments into, mapped to its corpus frequency.
TokenizedWordCounts = Counter  # Counter[Tuple[str, ...]]
# A merge rule: the two tokens that get concatenated.
Merge = Tuple[str, str]

UNK_TOKEN = "<unk>"

DEFAULT_MIN_FREQUENCY = 2


def find_missing_characters(word_counts: WordCounts, vocab: Set[str]) -> List[str]:
    """Return the characters occurring in the corpus that the vocabulary does not contain.

    A tokenizer without byte fallback turns an unknown character into ``<unk>``, which cannot be
    reversed, so these have to be added to the vocabulary before any text is tokenized.
    """
    charset = {char for word in word_counts for char in word}
    charset = {char.strip() for char in charset}
    charset.discard("")
    return sorted(charset - vocab)


def split_added_token_counts(shared: List[str], src_only: List[str], trg_only: List[str]) -> Tuple[int, int]:
    """Split added tokens into a per-side count for the tokenization stats.

    ``shared`` holds tokens that both sides drew from one pool and so cannot be attributed to
    either; they are divided evenly. ``src_only`` and ``trg_only`` must not also appear in
    ``shared``, or they would be counted twice.
    """
    # TODO: Calculate representative split of tokens for shared vocab case
    src_share = len(shared) // 2
    return len(src_only) + src_share, len(trg_only) + len(shared) - src_share


def normalize_merges(merges: Sequence) -> List[List[str]]:
    """Return ``merges`` as a list of ``[left, right]`` pairs.

    A ``tokenizer.json`` stores merges either as space-joined strings ("a n") or as pairs
    (["a", "n"]) depending on which version of the tokenizers library wrote it.  A file mixing
    both forms fails to deserialize, so the whole list has to be normalized before anything is
    appended to it.
    """
    normalized: List[List[str]] = []
    for merge in merges:
        if isinstance(merge, str):
            parts = merge.split(" ")
        elif isinstance(merge, (list, tuple)):
            parts = list(merge)
        else:
            raise ValueError(f"Unrecognized merge entry: {merge!r}")
        if len(parts) != 2:
            raise ValueError(f"Expected a merge of exactly two tokens, got: {merge!r}")
        normalized.append([parts[0], parts[1]])
    return normalized


def add_bpe_tokens_and_merges(
    tokenizer_json: dict, new_tokens: Sequence[str], new_merges: Sequence[Merge] = ()
) -> None:
    """Add ``new_tokens`` to the vocabulary and append ``new_merges`` after the existing rules.

    Mutates ``tokenizer_json`` in place. Appending leaves every existing merge outranking the new
    ones, so the base tokenizer runs to its fixed point before any of them fire. Every merge result
    must end up in the vocabulary, and the merge list must use a single encoding throughout, or the
    tokenizer will not load at all.
    """
    vocab: Dict[str, int] = tokenizer_json["model"]["vocab"]
    next_id = max(vocab.values()) + 1 if vocab else 0
    for token in new_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    merges = normalize_merges(tokenizer_json["model"]["merges"])
    merges.extend([left, right] for left, right in new_merges)
    tokenizer_json["model"]["merges"] = merges


def build_extended_bpe_tokenizer(
    tokenizer_json: dict, new_tokens: Sequence[str], new_merges: Sequence[Merge]
) -> Tuple[Tokenizer, dict]:
    """Load a copy of ``tokenizer_json`` with the given tokens and merges added.

    Returns the loaded tokenizer along with the JSON it was built from, so that callers can stage
    several rounds of additions without serializing the tokenizer back out in between.
    """
    data = deepcopy(tokenizer_json)
    add_bpe_tokens_and_merges(data, new_tokens, new_merges)
    return Tokenizer.from_str(json.dumps(data)), data


def tokenize_words(tokenizer: Tokenizer, word_counts: WordCounts) -> Tuple[TokenizedWordCounts, WordCounts]:
    """Tokenize each distinct word and return (counts by token sequence, rejected words).

    Segmenting with the real tokenizer rather than reimplementing BPE guarantees the learner sees
    exactly what the model will see at run time.  Words containing ``<unk>`` are rejected: their
    original characters cannot be recovered, so no useful merge can be learned from them.
    """
    tokenized: TokenizedWordCounts = Counter()
    rejected: WordCounts = Counter()
    for word, freq in word_counts.items():
        tokens = tuple(token.value for token in tokenizer.model.tokenize(word))
        if UNK_TOKEN in tokens:
            rejected[word] += freq
        else:
            tokenized[tokens] += freq
    return tokenized, rejected


def learn_bpe_merges(
    tokenized_words: TokenizedWordCounts,
    num_merges: int,
    vocab: Set[str],
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    forbidden_pairs: FrozenSet[Merge] = frozenset(),
) -> Tuple[List[Merge], List[str], List[int]]:
    """Learn up to ``num_merges`` BPE merges over the base tokens each word segments into.

    Returns the merges in learning order, the token each one produces, and how often each pair
    occurred when it was chosen.  Merge ``k`` may combine a token produced by an earlier merge, so
    the order is load-bearing and must be preserved when the rules are written out.

    Every returned merge produces exactly one new vocabulary entry: pairs whose concatenation is
    already known are skipped, so ``len(merges) == len(new_tokens)``.  Fewer than ``num_merges``
    come back when the corpus runs out of pairs occurring at least ``min_frequency`` times.
    """
    # One entry per distinct word, holding the tokens that word currently segments into. Merging
    # rewrites these entries in place, so they track the segmentation as it evolves.
    word_tokens: List[List[str]] = [list(tokens) for tokens in tokenized_words]
    freqs: List[int] = list(tokenized_words.values())

    pair_counts: Counter = Counter()
    pair_words: Dict[Merge, Set[int]] = defaultdict(set)
    for index, tokens in enumerate(word_tokens):
        freq = freqs[index]
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += freq
            pair_words[pair].add(index)

    known = set(vocab)
    blocked = set(forbidden_pairs)
    for pair in blocked.intersection(pair_counts):
        del pair_counts[pair]

    merges: List[Merge] = []
    new_tokens: List[str] = []
    firing_counts: List[int] = []
    while len(merges) < num_merges and pair_counts:
        best_count = max(pair_counts.values())
        if best_count < min_frequency:
            break
        # Break ties on the pair itself so that the result does not depend on iteration order.
        best = min(pair for pair, count in pair_counts.items() if count == best_count)
        left, right = best
        merged = left + right  # the new token this merge creates
        if merged in known:
            # Adding this merge would not add a token, and could demote an existing rule.
            blocked.add(best)
            del pair_counts[best]
            continue

        merges.append(best)
        new_tokens.append(merged)
        firing_counts.append(best_count)
        known.add(merged)

        for index in list(pair_words[best]):
            tokens = word_tokens[index]
            freq = freqs[index]
            updated: List[str] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                    updated.append(merged)
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            old_pairs = Counter(zip(tokens, tokens[1:]))
            fresh_pairs = Counter(zip(updated, updated[1:]))
            for pair, count in old_pairs.items():
                if pair in blocked or pair not in pair_counts:
                    continue
                pair_counts[pair] -= count * freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                if pair not in fresh_pairs:
                    pair_words[pair].discard(index)
            for pair, count in fresh_pairs.items():
                if pair in blocked:
                    continue
                pair_counts[pair] += count * freq
                pair_words[pair].add(index)
            word_tokens[index] = updated

        pair_counts.pop(best, None)
        pair_words.pop(best, None)

    return merges, new_tokens, firing_counts
