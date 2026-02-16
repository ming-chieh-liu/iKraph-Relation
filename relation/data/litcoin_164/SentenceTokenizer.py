from __future__ import annotations
import re
import logging
from typing import TypedDict, Optional
from collections.abc import Iterable

import nltk
from nltk.corpus import words

logger = logging.getLogger(__name__)


class SentenceTokenizeWithOffsets(TypedDict):
    """
    A dictionary with the following 3 fields:
    - text (str): sentence
    - offset_start (int): starting offset
    - offset_end (int): ending offset
    """

    text: str
    offset_start: int
    offset_end: int


def split_case_1(text: str) -> Optional[tuple[int, int]]:
    """
    This case is defined by the following pattern:
    1. a character: '.' or '?'
    2. followed by some whitespace characters
    3. followed by a word beginning with capitalized letter
    4. followed by a whitespace character.

    When there are multiple matching cases, the function only splits
    the first case. User can call the function iteratively to find all
    matching cases.

    The vanilla tokenizer can split the following case:
    'Job is done after 24h. Then we leave.'
    but not the following one:
    'Job is done after 24 h. Then we leave.'

    Output:
        In case the text won't be split, return None.
        Otherwise, returns a tuple indicating
        the ending position of the first splitted sentence,
        and the starting position of the second splitted sentence.

    Example:
        >>> text = 'Job is done after 24 h. Then we leave.'
        >>> end_pos, start_pos = split_case_1(text)
        >>> text[0: end_pos]
        'Job is done after 24 h.'
        >>> text[start_pos: ]
        'Then we leave.'
    """
    pattern = re.compile(r"([\.\?])\s([A-Z][a-z]*)\s")
    result = pattern.search(text)
    if result != None:
        # variable 'word' below is the captital lized word
        # we found after '.' or '?'.
        # The matching [A-Z][a-z]* in the regex above.
        word = result.group(2)
        if word != '' and word.lower() in words.words():
            # end position of group 1, the character '.' or '?'
            end_pos = result.end(1)
            start_pos = result.start(2)
            return (end_pos, start_pos)
    return None

def merge_case_1(prev_sent: SentenceTokenizeWithOffsets, next_sent: SentenceTokenizeWithOffsets, text: str) -> bool:
    """
    Merge when the text ends with ')'.

    The vanilla tokenizer would not split the following case:
    'Antithrombin (50 mg/day) Was administered.'
    But would split the following one:
    'Antithrombin (50 mg/i.v.) Was administered.'
    """
    return prev_sent["text"].endswith(')')


def merge_case_2(prev_sent: SentenceTokenizeWithOffsets, next_sent: SentenceTokenizeWithOffsets, text: str) -> bool:
    """
    Merge when the tokenizer splits a whole word into different sentences.

    The vanilla tokenizer would split the following sentence
    'Among analyzed variants, CACNG8c.*6819A>T showed low frequency'
    into
    'Among analyzed variants, CACNG8c.', '*6819A>T showed low frequency'
    """

    prev_end = prev_sent['offset_end']
    next_start = next_sent['offset_start']
    return prev_end == next_start


class LitcoinSentenceTokenizer:
    """
    Sentence tokenizer used for the Litcoin challenge.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extra_abbreviations = set(
            [
                "e.g",
                "i.e",
                "i.m",
                "a.u",
                "p.o",
                "i.v",
                "i.p",
                "vivo",
                "p.o",
                "i.p",
                "Vmax",
                "i.c.v",
                ")(",
                "E.C",
                "sp",
                "al",
            ]
        )
        self.nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.nltk_tokenizer._params.abbrev_types.update(self.extra_abbreviations)

        self.split_fns = [split_case_1]
        self.merge_fns = [merge_case_1, merge_case_2]

    def add_extra_abbreviations(
        self, abbreviations: Iterable[str], warn_trailing_dots: bool = True
    ) -> None:
        """Add extra abbreviations to the underlying NLTK tokenizer.

        Extra abbreviations are tokens that the tokenizer shouldn't split on.

        Example: Coding languages, e.g. C, is good for human race.
        This sentence shouldn't be split right after abbreviation e.g.

        Current abbreviations added can be accessed via
        `LitcoinSentenceTokenizer.extra_abbreviations`.
        Added abbreviations should not include the trailing dot.

        Example: To add e.g., one should only include e.g

        Args:
            abbreviations (Iterable[str]): An iterable (list, set, etc) of abbreviations
                to add into the tokenizer.
            warn_trailing_dots (bool): Warn users about abbreviations having a trailing dot.
                Defaults to True.
        """

        for abbreviation in abbreviations:
            if abbreviation.endswith(".") and warn_trailing_dots:
                logger.warn(
                    f"Found an abbreviation with a trailing dot: {abbreviation} ."
                    " Please read the documentation for details. You can suppress this"
                    " warning by adding `warn_trailing_dots=False`"
                )
            self.extra_abbreviations.add(abbreviation)

        self.nltk_tokenizer._params.abbrev_types.update(abbreviations)

    def sentence_tokenize(self, text: str) -> list[str]:
        # This tokenizer doesn't have a different method for splitting without getting offsets
        split_sentences_with_offsets = self.sentence_tokenize_with_offsets(text)
        sentences = [
            sentence_with_offset["text"]
            for sentence_with_offset in split_sentences_with_offsets
        ]
        return sentences

    def sentence_tokenize_with_offsets(
        self, text: str
    ) -> list[SentenceTokenizeWithOffsets]:
        sentences = self.nltk_tokenizer.tokenize(text)

        sentences_with_offsets_list = []
        prev_end = 0
        for sent in sentences:
            start = text.find(sent, prev_end)
            end = start + len(sent)
            sentences_with_offsets_list.append(
                SentenceTokenizeWithOffsets(
                    text=sent, offset_start=start, offset_end=end
                )
            )
            prev_end = end
        ret = self.post_processing(text, sentences_with_offsets_list)
        return ret

    def post_processing(self, text:str, sentences: list[SentenceTokenizeWithOffsets]) -> list[SentenceTokenizeWithOffsets]:

        def split_sentence(sentence: SentenceTokenizeWithOffsets) -> list[SentenceTokenizeWithOffsets]:
            is_split = False
            new_sentences: list[SentenceTokenizeWithOffsets] = []
            for split_fn in self.split_fns:
                split_result = split_fn(sentence["text"])
                if split_result is None:
                    new_sentences.append(sentence)
                    continue
                is_split = True
                sentence_start = sentence["offset_start"]
                sentence_end = sentence["offset_end"]
                first_sentence_end, second_sentence_start = split_result
                first_sentence_text = sentence["text"][0: first_sentence_end]
                second_sentence_text = sentence["text"][second_sentence_start: ]
                first_sentence_end += sentence_start
                second_sentence_start += sentence_start
                new_sentences.append(
                    SentenceTokenizeWithOffsets(
                        text=first_sentence_text,
                        offset_start=sentence_start,
                        offset_end=first_sentence_end
                    )
                )
                new_sentences.append(
                    SentenceTokenizeWithOffsets(
                        text=second_sentence_text,
                        offset_start=second_sentence_start,
                        offset_end=sentence_end
                    )
                )
                break  # only do one split_fn per sentence per iteration
            if is_split:
                ret: list[SentenceTokenizeWithOffsets] = []
                for new_sentence in new_sentences:
                    ret += split_sentence(new_sentence)
                return ret
            else:
                return new_sentences

        split_sentences: list[SentenceTokenizeWithOffsets] = []
        for sentence in sentences:
            split_sentences += split_sentence(sentence)

        merged_sentences: list[SentenceTokenizeWithOffsets] = []

        if len(split_sentences) < 2:
            return split_sentences

        curr_idx = 0
        merged_sentences.append(split_sentences[0])
        while curr_idx < len(split_sentences) - 1:
            prev_sent = merged_sentences[-1]
            next_sent = split_sentences[curr_idx + 1]
            is_merge = any([merge_fn(prev_sent, next_sent, text) for merge_fn in self.merge_fns])
            if is_merge:
                prev_start = prev_sent["offset_start"]
                next_end = next_sent["offset_end"]
                merged_sentences[-1] = SentenceTokenizeWithOffsets(
                    text=text[prev_start: next_end],
                    offset_start=prev_start,
                    offset_end=next_end
                )
            else:
                merged_sentences.append(next_sent)
            curr_idx += 1

        return merged_sentences

